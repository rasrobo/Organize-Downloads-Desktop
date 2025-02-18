import os
import logging
import json
import csv
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from guessit import guessit
from imdb import Cinemagoer as IMDb  # Updated import
import re
from rich.console import Console
from rich.table import Table
import subprocess
import hashlib
from collections import defaultdict

# Update FOLDER_MAPPINGS at top of file with AI patterns
FOLDER_MAPPINGS = {
    "downloads": {
        "destinations": {
            "images": "Pictures",
            "documents": "Documents",
            "videos": "Videos",
            "music": "Music",
            "ai_images": "Pictures/AI_Generated",
            "ai_video": "Videos/AI_Generated",
            "ai_music": "Music/AI_Generated"
        },
        "extensions": {
            "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"],
            "documents": [".pdf", ".doc", ".docx", ".txt", ".xlsx", ".csv"],
            "videos": [".mp4", ".mkv", ".avi", ".mov", ".wmv"],
            "music": [".mp3", ".wav", ".flac", ".m4a", ".ogg"]
        },
        "patterns": {
            "ai_images": ["midjourney", "stable_diffusion", "dall-e", "sd_", "mj_"],
            "ai_video": ["runway_", "gen1_", "pika_", "stable_video", "sv_"],
            "ai_music": ["suno_", "udio_", "musicgen_", "musiclm_", "mubert_"]
        },
        "ai_music_patterns": {
            "structural": [
                r'\[(verse|chorus|bridge|instrumental).*?\]',  # [Verse 1], [Chorus]
                r'\[.*?(section|arrangement).*?\]',           # [Section], [Arrangement]
                r'\[.*?(fade|volume|intensity).*?\]',        # [Fade in/out]
                r'\[.*?(instrument|percussion|melody).*?\]',  # [Instruments]
                r'\((x\d+|\d+x)\)'                          # (x3), (3x)
            ],
            "version": [
                r'v\d+\.\d+\.\d+',                          # v1.2.3
                r'ext.*v\d+',                               # ext_v1
                r'iteration.*\d+',                          # iteration_1
                r'gen.*v\d+',                              # gen_v1
            ]
        }
    }
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
console = Console()

class FileOrganizer:
    def __init__(self, config: Dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.stats = {
            'total': 0,
            'success': 0,
            'errors': 0,
            'destinations': {},
            'duplicates': 0,
            'skipped': 0  # Initialize skipped counter
        }
        self.moved_files = []
        self.file_hashes = defaultdict(list)  # SHA-256 hash -> list of file paths
        
        # Define extensions to keep in original location
        self.keep_in_place_extensions = {
            # Code files
            ".py", ".js", ".html", ".css", ".java", ".cpp", ".h", ".php",
            # Config files
            ".json", ".yaml", ".yml", ".toml", ".ini",
            # Git files
            ".gitignore", ".gitattributes",
            # Virtual environment
            ".virtualenv", ".venv",
            # Project files
            "requirements.txt", "package.json", "composer.json",
            # Python cache files
            ".pyc", ".pyo", ".pyd",
            # Version control
            ".git", ".svn"
        }

    def detect_ai_generated(self, file_path: Path) -> Optional[str]:
        """Detect if file is AI-generated and return the AI tool used"""
        filename = file_path.name.lower()
        ai_patterns = {
            'midjourney': ['mj_', 'midjourney', '_mj'],
            'stable_diffusion': ['sd_', 'stable_diffusion'],
            'dalle': ['dalle_', 'dall-e'],
            'runway': ['runway_', 'rv_'],
            'pika': ['pika_', 'pika_labs'],
            'suno': ['suno_', 'bark_'],
            'musicgen': ['musicgen_', 'mg_'],
        }

        for tool, patterns in ai_patterns.items():
            if any(pattern in filename for pattern in patterns):
                return tool
        return None

    def detect_and_rename_media(self, file_path: Path) -> Optional[str]:
        """Detect media information and return formatted name"""
        try:
            guess = guessit(str(file_path))
            if not guess.get('title'):
                return None

            ia = IMDb()
            movies = ia.search_movie(guess['title'])
            if not movies:
                return None

            movie = movies[0]
            ia.update(movie)

            title = movie.get('title', '')
            year = movie.get('year', '')
            genres = movie.get('genres', [])

            is_family = bool(set(genres) & {'Family', 'Animation', 'Fantasy'})
            base_name = re.sub(r'[^\w\s-]', '', title).replace(' ', '_')
            
            formatted_name = f"{'FAMILY-' if is_family else ''}{base_name}_{year}"
            logger.info(f"Media detected: {formatted_name} (IMDb: https://www.imdb.com/title/tt{movie.movieID}/)")
            return formatted_name
        except Exception as e:
            logger.error(f"Error detecting media info: {e}")
            return None

    def compute_file_sha256(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file in chunks."""
        sha256 = hashlib.sha256()
        try:
            with file_path.open("rb") as f:
                for block in iter(lambda: f.read(65536), b""):
                    sha256.update(block)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return ""

    def _detect_file_category(self, file_ext: str) -> str:
        """Detect appropriate category for file extension"""
        # Common file type mappings
        type_mappings = {
            # Images
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.jfif', '.tiff', '.ico'],
            # Documents
            'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.xlsx', '.csv', '.pptx'],
            # Audio
            'music': ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'],
            # Video
            'video': ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.webm', '.flv', '.m4v'],
            # Archives
            'archives': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'],
            # Code
            'code': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.h', '.php'],
            # Executables
            'executables': ['.exe', '.msi', '.deb', '.rpm', '.app', '.dmg'],
            # System
            'system': ['.sys', '.dll', '.so', '.dylib', '.reg', '.ini', '.config', '.pat']
        }

        # Check known mappings
        for category, extensions in type_mappings.items():
            if file_ext.lower() in extensions:
                return category

        # If no match found, use MIME type detection as fallback
        try:
            import magic
            mime = magic.Magic(mime=True)
            mime_type = mime.from_buffer(open(file_path, 'rb').read(2048))
            main_type = mime_type.split('/')[0]
            
            # Map MIME main types to categories
            mime_mappings = {
                'image': 'images',
                'audio': 'music',
                'video': 'video',
                'text': 'documents',
                'application': 'executables'
            }
            return mime_mappings.get(main_type, 'other')
        except ImportError:
            # If python-magic not available, use extension as category
            return file_ext[1:].lower()  # Remove leading dot

    def process_file(self, file_path: Path) -> None:
        """Process a file and move its entire parent folder if it contains media files."""
        try:
            self.stats['total'] += 1
            file_ext = file_path.suffix.lower()

            # Define base destination directory
            base_dest = Path("/mnt/z/sort")

            # Skip processing if file should stay in place
            if file_ext in self.keep_in_place_extensions or file_path.name in self.keep_in_place_extensions:
                if file_ext in ['.py', '.pyc', '.pyd']:  # Development files
                    logger.debug(f"Skipping development file: {file_path.name}")
                else:
                    logger.debug(f"Keeping in place: {file_path.name}")
                self.stats['skipped'] += 1
                return

            # Media type groups and their extensions
            media_groups = {
                "Videos": {
                    "main": [".mkv", ".mp4", ".avi", ".mov", ".webm"],
                    "related": [".srt", ".sub", ".idx", ".nfo", ".txt", ".jpg", ".png"]
                },
                "Music": {
                    "main": [".mp3", ".wav", ".ogg", ".m4a", ".flac"],
                    "related": [".cue", ".log", ".m3u", ".jpg", ".png", ".txt"]
                },
                "Pictures": {
                    "main": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"],
                    "related": [".txt", ".md"]
                },
                "Documents": {
                    "main": [".pdf", ".doc", ".docx", ".txt", ".xlsx", ".csv"],
                    "related": [".jpg", ".png"]
                }
            }

            # Find the media group this file belongs to
            for group_name, extensions in media_groups.items():
                if file_ext in extensions["main"]:
                    source_folder = file_path.parent
                    folder_name = source_folder.name

                    # Check if folder contains related files
                    related_files = [
                        f for f in source_folder.iterdir()
                        if f.is_file() and f != file_path and 
                        f.suffix.lower() in extensions["related"]
                    ]

                    if related_files:
                        # Move the entire folder
                        dest_folder = base_dest / group_name / folder_name
                        if not self.dry_run:
                            dest_folder.mkdir(parents=True, exist_ok=True)
                            for item in source_folder.iterdir():
                                new_path = dest_folder / item.name
                                item.rename(new_path)
                                logger.info(f"Moving: {item.name} → {new_path}")
                                self.moved_files.append((str(item), str(new_path)))
                            self.stats['success'] += 1
                        else:
                            logger.info(f"Would move folder {folder_name} to {group_name}/")
                            for item in source_folder.iterdir():
                                logger.debug(f"  - {item.name}")
                    else:
                        # Move single file
                        dest_path = base_dest / group_name / file_path.name
                        if not self.dry_run:
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            file_path.rename(dest_path)
                            logger.info(f"Moving: {file_path.name} → {dest_path}")
                            self.moved_files.append((str(file_path), str(dest_path)))
                            self.stats['success'] += 1
                        else:
                            logger.info(f"Would move: {file_path.name} → {group_name}/")
                    break

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            self.stats['errors'] += 1

    def _get_media_type(self, extension: str) -> Optional[str]:
        """Helper method to determine media type from extension"""
        extension = extension.lower()
        media_types = {
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'],
            'videos': ['.mp4', '.mkv', '.avi', '.mov', '.wmv'],
            'music': ['.mp3', '.wav', '.flac', '.m4a', '.ogg'],
            'documents': ['.pdf', '.doc', '.docx', '.txt', '.xlsx', '.csv']
        }
        
        for media_type, extensions in media_types.items():
            if extension in extensions:
                return media_type.rstrip('s')  # Remove plural 's' for consistency
        return None

    def generate_report(self, format: str = 'text', report_dir: Optional[str] = None) -> None:
        """Generate report in specified format"""
        if report_dir:
            report_path = Path(report_dir)
        else:
            report_path = Path.home() / 'Documents' / 'odd_reports'
        
        report_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            report_file = report_path / f'odd_report_{timestamp}.json'
            with open(report_file, 'w') as f:
                json.dump({
                    'stats': self.stats,
                    'moved_files': self.moved_files
                }, f, indent=2)
        elif format == 'csv':
            report_file = report_path / f'odd_report_{timestamp}.csv'
            with open(report_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Source', 'Destination'])
                writer.writerows(self.moved_files)
        elif format == 'html':
            report_file = report_path / f'odd_report_{timestamp}.html'
            # Implementation for HTML report
        else:  # text format
            self._print_text_summary()

    def _print_text_summary(self) -> None:
        """Print text summary"""
        console.print(f"Total Files Processed: {self.stats['total']}", style="bold blue")
        console.print(f"Successfully Moved: {self.stats['success']}", style="bold green")
        console.print(f"Errors: {self.stats['errors']}", style="bold red")
        console.print(f"Duplicates Removed: {self.stats['duplicates']}", style="bold yellow")
        
        console.print("\nDuplicate Files:", style="bold blue")
        for file_hash, paths in self.file_hashes.items():
            console.print(f"\nHash: {file_hash}", style="bold")
            console.print(f"Original: {paths[0]}")
            for dup in paths[1:]:
                console.print(f"Duplicate: {dup}", style="dim")
        
        console.print("\nNew File Locations:", style="bold blue")
        for idx, (source, dest) in enumerate(self.moved_files, 1):
            console.print(f"{idx}. {Path(source).name}")
            console.print(f"   → {dest}", style="dim")

    def get_video_metadata(self, file_path: Path) -> Dict[str, str]:
        """Extract metadata from video file using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                metadata = json.loads(result.stdout)
                # Extract useful metadata
                info = {
                    'title': '',
                    'resolution': '',
                    'duration': '',
                    'fps': '',
                    'codec': ''
                }
                
                # Get video stream info
                for stream in metadata.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        width = stream.get('width', '')
                        height = stream.get('height', '')
                        if width and height:
                            info['resolution'] = f"{width}x{height}"
                        
                        # Get FPS
                        fps = stream.get('r_frame_rate', '')
                        if fps and '/' in fps:
                            num, den = map(int, fps.split('/'))
                            info['fps'] = f"{round(num/den)}fps"
                        
                        info['codec'] = stream.get('codec_name', '')
                        break
                
                # Get duration
                if 'format' in metadata:
                    duration = float(metadata['format'].get('duration', 0))
                    if duration > 0:
                        duration_obj = timedelta(seconds=int(duration))
                        info['duration'] = str(duration_obj).split('.')[0]
                    
                    # Try to get title from metadata
                    tags = metadata['format'].get('tags', {})
                    info['title'] = tags.get('title', '')
                
                return info
        except Exception as e:
            logger.debug(f"Error extracting metadata: {e}")
        return {}

    def generate_descriptive_filename(self, file_path: Path, metadata: Dict[str, str]) -> str:
        """Generate descriptive filename from metadata"""
        base_name = file_path.stem
        extension = file_path.suffix

        # Try to use original name if it's not just a UUID/random string
        if not (base_name.count('-') >= 4 and len(base_name) >= 32):
            new_name = base_name
        else:
            new_name = 'video'
        
        # Add metadata details
        details = []
        if metadata.get('resolution'):
            details.append(metadata['resolution'])
        if metadata.get('fps'):
            details.append(metadata['fps'])
        if metadata.get('duration'):
            details.append(metadata['duration'])
        
        if details:
            new_name = f"{new_name}_{'_'.join(details)}"
        
        return f"{new_name}{extension}"

    def extract_ai_video_metadata(self, file_path: Path) -> Dict[str, str]:
        """Extract metadata from AI-generated video files"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                '-show_chapters',
                str(file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                metadata = json.loads(result.stdout)
                format_tags = metadata.get('format', {}).get('tags', {})
                
                # Common metadata fields used by AI video tools
                info = {
                    'model': format_tags.get('model', ''),
                    'prompt': format_tags.get('prompt', ''),
                    'negative_prompt': format_tags.get('negative_prompt', ''),
                    'seed': format_tags.get('seed', ''),
                    'steps': format_tags.get('steps', ''),
                    'guidance_scale': format_tags.get('cfg_scale', ''),
                    'motion_bucket': format_tags.get('motion_bucket_id', ''),
                    'tool': ''
                }
                
                # Detect AI tool from metadata or filename patterns
                filename = file_path.name.lower()
                if any(p in filename for p in ['pika_', 'pika-']):
                    info['tool'] = 'pika'
                elif any(p in filename for p in ['runway_', 'rv_']):
                    info['tool'] = 'runway'
                elif any(p in filename for p in ['gen1_', 'g1_']):
                    info['tool'] = 'gen1'
                elif any(p in filename for p in ['sv_', 'stable_video']):
                    info['tool'] = 'stable_video'
                    
                return info
        except Exception as e:
            logger.debug(f"Error extracting AI video metadata: {e}")
        return {}

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to be POSIX-compliant and human readable
        """
        # Remove file extension if present
        name_parts = filename.rsplit('.', 1)
        name = name_parts[0]
        extension = f".{name_parts[1]}" if len(name_parts) > 1 else ""
        
        # Replace common patterns
        replacements = {
            # AI tool markers
            "AI_Test_Kitchen_": "ATK_",
            "DALL·E": "DALLE",
            "stable_diffusion": "SD",
            "midjourney": "MJ",
            
            # Common words to abbreviate
            "_minimal_": "_min_",
            "_tropical_": "_trop_",
            "_instruments_": "_inst_",
            "_background_": "_bg_",
            "_soundtrack_": "_ost_",
                
            # Remove redundant separators
            "__": "_",
            "--": "-",
            "._": "_",
            
            # Common music terms
            "_orchestral_": "_orch_",
            "_electronic_": "_elec_",
            "_acoustic_": "_acou_"
        }
        
        # Apply replacements
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        # Remove any remaining problematic characters
        name = re.sub(r'[^\w\-\.]', '_', name)
        
        # Truncate if too long (leaving room for counter and extension)
        max_length = 255 - len(extension)
        if len(name) > max_length:
            name = name[:max_length]
        
        # Remove trailing separators
        name = name.rstrip('_-')
        
        return f"{name}{extension}"

    def generate_ai_video_filename(self, file_path: Path, metadata: Dict[str, str]) -> str:
        """Generate descriptive filename for AI-generated video"""
        base_name = file_path.stem
        extension = file_path.suffix
        
        # Start with the AI tool name if available
        parts = []
        if metadata.get('tool'):
            parts.append(metadata['tool'].upper())
        
        # Add prompt if available (truncated)
        if metadata.get('prompt'):
            prompt = metadata['prompt'][:50].replace(' ', '_')
            parts.append(prompt)
                
        # Add technical details if available
        if metadata.get('motion_bucket'):
            parts.append(f"motion{metadata['motion_bucket']}")
        if metadata.get('seed'):
            parts.append(f"seed{metadata['seed']}")
        
        # If no metadata available, keep original name
        if not parts:
            return file_path.name
            
        # Combine parts and ensure valid filename
        new_name = '-'.join(parts)
        new_name = re.sub(r'[^\w\-\.]', '_', new_name)
        return self.sanitize_filename(f"{new_name}{extension}")

    def extract_ai_audio_metadata(self, file_path: Path) -> Dict[str, str]:
        """Extract metadata from AI-generated audio files using enhanced detection."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                '-show_chapters',
                str(file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                metadata = json.loads(result.stdout)
                format_tags = metadata.get('format', {}).get('tags', {})
                
                info = {
                    'title': format_tags.get('title', ''),
                    'artist': format_tags.get('artist', ''),
                    'album': format_tags.get('album', ''),
                    'genre': format_tags.get('genre', ''),
                    'encoder': format_tags.get('encoder', ''),
                    'lyrics': format_tags.get('lyrics-eng', ''),
                    'comment': format_tags.get('comment', ''),
                    'software': format_tags.get('software', ''),
                    'creator': format_tags.get('TEXT', '') or format_tags.get('artist', ''),
                    'tool': '',
                    'is_ai': False
                }
                
                # Priority 1: Check for specific tool patterns in filename/title.
                filename = file_path.name.lower()
                title_text = info['title'].lower() if info['title'] else filename
                udio_patterns = [
                    (r'v\d+\.\d+\.\d+', 'Version number with dots'),
                    (r'ext.*v\d+\.\d+', 'Extended version'),
                    (r'remix.*v\d+\.\d+', 'Remix version'),
                    (r'udio.*v\d+', 'Udio version')
                ]
                suno_patterns = [
                    (r'iteration.*\d+', 'Iteration series'),
                    (r'echo.*\d+', 'Echo series'),
                    (r'cipher.*\d+', 'Cipher series'),
                    (r'\s\d+(\s*-\s*instrumental)?$', 'Numbered instrumental')
                ]
                for pattern, desc in udio_patterns:
                    if any(re.search(pattern, text) for text in [filename, title_text]):
                        info['is_ai'] = True
                        info['tool'] = 'udio'
                        logger.debug(f"Udio pattern detected ({desc}): {pattern}")
                        return info
                for pattern, desc in suno_patterns:
                    if any(re.search(pattern, text) for text in [filename, title_text]):
                        info['is_ai'] = True
                        info['tool'] = 'suno'
                        logger.debug(f"Suno pattern detected ({desc}): {pattern}")
                        return info

                # Priority 1.5: Check software and comment fields for AI tool markers.
                if info['software']:
                    sw = info['software'].lower()
                    if 'udio' in sw:
                        info['is_ai'] = True
                        info['tool'] = 'udio'
                        logger.debug(f"AI detected in software field: {info['software']}")
                        return info
                    elif 'suno' in sw:
                        info['is_ai'] = True
                        info['tool'] = 'suno'
                        logger.debug(f"AI detected in software field: {info['software']}")
                        return info
                if info['comment']:
                    comm = info['comment'].lower()
                    if 'udio' in comm:
                        info['is_ai'] = True
                        info['tool'] = 'udio'
                        logger.debug(f"AI detected in comment field: {info['comment']}")
                        return info
                    elif 'suno' in comm:
                        info['is_ai'] = True
                        info['tool'] = 'suno'
                        logger.debug(f"AI detected in comment field: {info['comment']}")
                        return info

                # Priority 2: Check artist field for AI tool markers.
                if info['artist']:
                    artist = info['artist'].lower()
                    if re.search(r'\budio\b', artist):
                        info['is_ai'] = True
                        info['tool'] = 'udio'
                        logger.debug(f"AI detected in artist field: {info['artist']}")
                        return info
                    elif re.search(r'\bsuno\b', artist):
                        info['is_ai'] = True
                        info['tool'] = 'suno'
                        logger.debug(f"AI detected in artist field: {info['artist']}")
                        return info

                # Priority 3: Check the creator field for generic AI indications.
                if info['creator']:
                    creator = info['creator'].lower()
                    ai_creator_patterns = [
                        r'\b(?:suno|bark|udio)\b',
                        r'ai[-_\s]?generated'
                    ]
                    if any(re.search(pattern, creator) for pattern in ai_creator_patterns):
                        info['is_ai'] = True
                        info['tool'] = 'udio' if re.search(r'v\d+\.\d+\.\d+', title_text) else 'suno'
                        logger.debug(f"AI creator field detected: {info['creator']}")
                        return info

                # Priority 4: Check for structural patterns in lyrics.
                lyrics = info['lyrics'].lower()
                if lyrics:
                    structural_patterns = [
                        r'\[(verse|chorus|bridge).*?\].*?\[',
                        r'\[.*?background.*?vocals.*?\]',
                        r'\[.*?instruments?.*?join.*?\]',
                        r'\[.*?fade.*?(in|out).*?\]'
                    ]
                    for pattern in structural_patterns:
                        if re.search(pattern, lyrics, re.MULTILINE | re.IGNORECASE):
                            info['is_ai'] = True
                            info['tool'] = 'suno'
                            logger.debug(f"AI structural pattern detected in lyrics: {pattern}")
                            return info
                    ai_lyrics_tags = ['[genre:', '[bpm:', '[key:', '[mood:', '[instruments:', '[style:']
                    if all(tag in lyrics for tag in ai_lyrics_tags[:3]) or any(tag in lyrics for tag in ai_lyrics_tags):
                        info['is_ai'] = True
                        info['tool'] = 'suno'
                        logger.debug("Multiple structured metadata tags detected in lyrics (GENRE, BPM, etc.)")
                        return info

                # Priority 5: Check global AI music patterns defined in configuration.
                global_patterns = FOLDER_MAPPINGS.get('downloads', {}).get('ai_music_patterns', {})
                if global_patterns:
                    for pattern in global_patterns.get('version', []):
                        if re.search(pattern, filename):
                            info['is_ai'] = True
                            info['tool'] = 'udio'
                            logger.debug(f"Global AI music version pattern detected: {pattern}")
                            return info
                    for pattern in global_patterns.get('structural', []):
                        if re.search(pattern, filename, re.IGNORECASE):
                            info['is_ai'] = True
                            info['tool'] = 'suno'
                            logger.debug(f"Global AI music structural pattern detected: {pattern}")
                            return info

                # NEW HEURISTIC 1: If nearly all metadata fields are empty and the stem ends with a number,
                # mark it as AI-generated with the default tool.
                if (not info['title'] and not info['artist'] and not info['lyrics'] and 
                    not info['comment'] and not info['software'] and not info['creator']):
                    stem = file_path.stem
                    if re.search(r'\d+$', stem):
                        info['is_ai'] = True
                        info['tool'] = 'suno'
                        logger.debug("Sparse metadata with numeric filename suffix detected - marked as AI (suno)")
                
                # NEW HEURISTIC 2: If title and artist are empty but lyrics are long, flag as AI-generated.
                if (not info['title'] and not info['artist']) and lyrics and len(lyrics) > 200:
                    info['is_ai'] = True
                    info['tool'] = 'suno'
                    logger.debug("Long lyrics with sparse title and artist detected - marked as AI (suno)")

                logger.debug(f"Final metadata for {file_path.name}: {format_tags}")
                logger.debug(f"Detected tool: {info['tool']} | is_ai: {info['is_ai']}")
                return info

        except Exception as e:
            logger.debug(f"Error extracting AI audio metadata from {file_path.name}: {e}")
        return {}

    def generate_audio_filename(self, file_path: Path, metadata: Dict[str, str]) -> str:
        """Generate a standardized filename from audio metadata"""
        # Extract components
        creator = metadata.get('TEXT', '') or metadata.get('artist', '') or file_path.stem
        title = metadata.get('title', '') or file_path.stem

        # Clean up components
        parts = []
        if creator:
            parts.append(creator.replace(' ', '_'))
        parts.append(title)

        # Extract a version number if present and remove it from the title component
        version_match = re.search(r'v?(\d+(?:\.\d+)*)', title)
        version_component = ""
        if version_match:
            version_component = version_match.group(1)
            parts[-1] = re.sub(r'v?\d+(?:\.\d+)*', '', parts[-1]).strip()

        # Reassemble filename from parts
        new_filename = '-'.join(filter(None, parts))
        if version_component:
            new_filename = f"{new_filename}-v{version_component}"
        new_filename = f"{new_filename}{file_path.suffix}"

        # If the file is detected as AI-generated then prepend the tool name
        if metadata.get('is_ai'):
            tool = metadata.get('tool', 'UNKNOWN_AI').upper()
            new_filename = f"{tool}_{new_filename}"

        logger.debug(f"Generated new audio filename: {new_filename}")
        return new_filename

    def process_audio_file(self, file_path: Path, metadata: Dict[str, str]) -> Path:
        """Process audio file and determine destination path"""
        new_name = self.generate_audio_filename(file_path, metadata)
        # Determine base path based on AI detection
        if metadata.get('is_ai'):
            tool = metadata.get('tool', 'UNKNOWN_AI').upper()
            base_path = file_path.parent / 'Music/AI_Generated' / tool
        else:
            base_path = file_path.parent / 'Music'
        
        # Generate new filename
        new_path = base_path / new_name
        logger.debug(f"Filename transformation: {file_path.name} -> {new_name}")
        logger.debug(f"Metadata used: {metadata}")
        return new_path

    def process_directory(self, dir_path: Path, merge: bool = False) -> None:
        """Process directory recursively with optional subfolder merging"""
        try:
            # Skip processing packages and special directories
            skip_patterns = {
                # Python-specific
                'site-packages', 'dist-packages', '__pycache__',
                'venv', 'env', '.venv', '.env',
                # Version control
                '.git', '.svn',
                # IDE
                '.idea', '.vscode',
                # System
                '.local', '.cache'
            }

            # Skip if directory matches any skip pattern
            if any(part in skip_patterns or part.startswith('.') or part.endswith('.dist-info')
                   for part in dir_path.parts):
                logger.debug(f"Skipping special directory: {dir_path}")
                return

            # Skip if directory contains package indicators
            if any(dir_path.glob(p) for p in ['setup.py', 'pyproject.toml', 'requirements.txt']):
                logger.debug(f"Skipping Python package directory: {dir_path}")
                return

            # Process all files in current directory
            for file_path in dir_path.glob('*'):
                if file_path.is_file():
                    # Skip package metadata files
                    if file_path.name.lower() in {
                        'license', 'license.txt', 'license.md',
                        'readme', 'readme.txt', 'readme.md',
                        'setup.py', 'setup.cfg', 'pyproject.toml',
                        'requirements.txt', 'manifest.in',
                        'entry_points.txt', 'top_level.txt'
                    }:
                        continue
                    
                    self.process_file(file_path)
                elif file_path.is_dir():
                    # Skip if directory name matches skip patterns
                    if (file_path.name in skip_patterns or
                        file_path.name.startswith('.') or
                        file_path.name.endswith('.dist-info')):
                        continue
                    
                    # Recursively process valid subdirectories
                    self.process_directory(file_path, merge)
                    
                    # Remove empty directories if merge is enabled
                    if merge and file_path.exists():
                        try:
                            if not any(file_path.iterdir()):
                                file_path.rmdir()
                                logger.debug(f"Removed empty directory: {file_path}")
                        except Exception as e:
                            logger.error(f"Error removing directory {file_path}: {e}")

        except Exception as e:
            logger.error(f"Error processing directory {dir_path}: {e}")
            self.stats['errors'] += 1

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Organize files in directories.')
    parser.add_argument('-s', '--source', type=str, help='Source directory')
    parser.add_argument('-c', '--config', type=str, help='Config file path')
    parser.add_argument('-d', '--dry-run', action='store_true', help='Dry run')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--summary', action='store_true', help='Show detailed summary')
    parser.add_argument('--report', choices=['text', 'json', 'csv', 'html'], default='text')
    parser.add_argument('--report-dir', type=str, help='Custom report directory')
    parser.add_argument('--ai-only', action='store_true', help='Process only AI-generated files')
    parser.add_argument('-r', '--recursive', action='store_true', 
                       help='Process directories recursively')
    parser.add_argument('-m', '--merge', action='store_true',
                       help='Merge processed subfolders with parent')
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    # Load configuration
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = FOLDER_MAPPINGS
        
    source_dir = Path(args.source) if args.source else Path.home() / 'Downloads'
    organizer = FileOrganizer(config, args.dry_run)
    
    if args.recursive:
        organizer.process_directory(source_dir, args.merge)
    else:
        # Original non-recursive behavior
        for file_path in source_dir.glob('*'):
            if file_path.is_file():
                organizer.process_file(file_path)
                
    if args.summary:
        organizer.generate_report(args.report, args.report_dir)

if __name__ == "__main__":
    main()