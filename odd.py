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
import time  # Import time module
import pickle

# Update FOLDER_MAPPINGS at top of file with AI patterns
FOLDER_MAPPINGS = {
    "downloads": {
        "destinations": {
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
        self.source_dir = None  # Add source_dir attribute
        self.stats = {
            'total': 0,
            'success': 0,
            'errors': 0,
            'destinations': {},
            'duplicates': 0,
            'skipped': 0
        }
        self.moved_files = []
        self.file_hashes = defaultdict(list)
        self.processed_files = set()
        self.dest_base = Path("/mnt/z/sort")
        self.cache_dir = Path.home() / '.odd' / 'cache'
        self.cache_file = self.cache_dir / 'processed_files.pkl'
        self.imdb_cache_file = self.cache_dir / 'imdb_cache.pkl'
        self.processed_cache = self._load_cache()
        self.imdb_cache = self._load_imdb_cache()
        self.cache_modified = False  # Add this line
        self.last_cache_save = time.time()  # Add this line

    def _load_cache(self) -> Dict[str, dict]:
        """Load cache of processed files with metadata"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, 'rb') as f:
                        cache = pickle.load(f)
                        # Validate cache format
                        if isinstance(cache, dict):
                            return cache
                        logger.warning("Invalid cache format, creating new cache")
                        return {}
                except (EOFError, pickle.UnpicklingError) as e:
                    logger.warning(f"Cache file corrupted, creating new cache: {e}")
                    return {}
            return {}
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return {}

    def _load_imdb_cache(self) -> Dict[str, dict]:
        """Load IMDb lookup cache"""
        try:
            if self.imdb_cache_file.exists():
                with open(self.imdb_cache_file, 'rb') as f:
                    return pickle.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading IMDb cache: {e}")
            return {}

    def _save_cache(self) -> None:
        """Save cache with periodic cleanup"""
        try:
            # Only save if cache was modified and at least 5 minutes have passed
            current_time = time.time()
            if not self.cache_modified or (current_time - self.last_cache_save) < 300:
                return

            # Initialize new cache structure
            current_cache = {}
            cutoff = datetime.now() - timedelta(days=30)

            for cache_key, cache_data in self.processed_cache.items():
                try:
                    # Verify cache entry structure
                    if not isinstance(cache_data, dict):
                        logger.debug(f"Skipping invalid cache entry format: {cache_key}")
                        continue
                        
                    # Ensure all required fields are present
                    required_fields = {'path', 'category', 'size', 'mtime', 'processed_date'}
                    if not all(field in cache_data for field in required_fields):
                        logger.debug(f"Skipping cache entry missing fields: {cache_key}")
                        continue

                    # Parse and validate date
                    try:
                        processed_date = datetime.fromisoformat(cache_data['processed_date'])
                    except (ValueError, TypeError):
                        logger.debug(f"Invalid date format in cache entry: {cache_key}")
                        continue

                    # Keep only recent entries
                    if processed_date > cutoff:
                        current_cache[cache_key] = {
                            'path': str(cache_data['path']),
                            'category': str(cache_data['category']),
                            'size': int(cache_data['size']),
                            'mtime': float(cache_data['mtime']),
                            'processed_date': cache_data['processed_date']
                        }

                except (KeyError, ValueError, TypeError) as e:
                    logger.debug(f"Invalid cache entry data: {e}")
                    continue

            # Save cleaned cache
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(current_cache, f)

            # Update in-memory cache and reset flags
            self.processed_cache = current_cache
            self.cache_modified = False
            self.last_cache_save = current_time
            logger.debug(f"Cache saved with {len(current_cache)} entries")

        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _save_imdb_cache(self) -> None:
        """Save IMDb lookup cache"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.imdb_cache_file, 'wb') as f:
                pickle.dump(self.imdb_cache, f)
        except Exception as e:
            logger.error(f"Error saving IMDb cache: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Get quick hash of file (first 1MB only for speed)"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                chunk = f.read(1024 * 1024)  # Read first 1MB only
                hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return ""

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
            # Clean up the name for better IMDb matching
            clean_name = file_path.name
            # Remove common patterns from release names
            patterns = [
                r'\.(?:720p|1080p|2160p|DVDRip|BDRip|BluRay|WEB-DL|WEBDL|WEBRip)',
                r'\.(?:x264|x265|XviD|H264|H\.264|HEVC)',
                r'(?:-|\.)?(?:RETRO|PHOENiX|LPD|tRuAVC)',
                r'\.[0-9]{4}',  # Year
                r'\.3D',
                r'\.DVD'
            ]
            
            for pattern in patterns:
                clean_name = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
            
            # Replace dots and underscores with spaces
            clean_name = clean_name.replace('.', ' ').replace('_', ' ').strip()
            logger.debug(f"Cleaned name for IMDb search: {clean_name}")

            ia = IMDb()
            movies = ia.search_movie(clean_name)
            if not movies:
                logger.debug(f"No IMDb results for: {clean_name}")
                return None

            movie = movies[0]
            ia.update(movie)
            logger.debug(f"Found movie: {movie.get('title')} ({movie.get('year')})")

            title = movie.get('title', '')
            year = movie.get('year', '')
            genres = movie.get('genres', [])

            # Improved family content detection
            family_keywords = {'Family', 'Animation', 'Fantasy', 'Adventure'}
            genre_set = set(g.lower() for g in genres)
            is_family = (
                bool(genre_set & {g.lower() for g in family_keywords}) or
                any(k.lower() in title.lower() for k in ['disney', 'pixar', 'dreamworks'])
            )

            logger.debug(f"Genres: {genres}")
            logger.debug(f"Is family content: {is_family}")

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

    def find_matching_media_folder(self, sfv_path: Path, base_dest: Path) -> Optional[Path]:
        """Find matching media folder for an SFV file based on similar name patterns."""
        try:
            # Clean up the SFV folder name for matching
            sfv_folder_name = sfv_path.parent.name.lower()
            # Remove common release tags
            clean_name = re.sub(r'[\.\-](RETRO|xvid|x264|720p|1080p|bluray|bdrip|dvdrip)', '', 
                              sfv_folder_name, flags= re.IGNORECASE)
            
            # Look in Videos folder first
            videos_dir = base_dest / "Videos"
            if videos_dir.exists():
                for folder in videos_dir.iterdir():
                    if folder.is_dir():
                        folder_clean = re.sub(r'[\.\-](RETRO|xvid|x264|720p|1080p|bluray|bdrip|dvdrip)', '',
                                            folder.name.lower(), flags=re.IGNORECASE)
                        # Compare cleaned names
                        if clean_name.startswith(folder_clean) or folder_clean.startswith(clean_name):
                            return folder

            # Also check root level for similar named folders
            for folder in base_dest.iterdir():
                if folder.is_dir() and folder.name.lower() not in {'videos', 'music', 'pictures', 'documents'}:
                    folder_clean = re.sub(r'[\.\-](RETRO|xvid|x264|720p|1080p|bluray|bdrip|dvdrip)', '',
                                        folder.name.lower(), flags=re.IGNORECASE)
                    if clean_name.startswith(folder_clean) or folder_clean.startswith(clean_name):
                        return folder

            return None
        except Exception as e:
            logger.error(f"Error finding matching media folder: {e}")
            return None

    def find_media_home(self, file_path: Path, base_dest: Path) -> Optional[Path]:
        """Find matching media directory for a file based on name similarity."""
        try:
            # Skip if file is in keep_in_place list
            if file_path.suffix.lower() in self.keep_in_place_extensions:
                return None

            # Clean up the source name for matching
            source_name = file_path.parent.name.lower()
            
            # Skip generic directory names
            generic_dirs = {'downloads', 'videos', 'music', 'pictures', 'documents'}
            if source_name in generic_dirs:
                return None
                
            clean_source = re.sub(
                r'[\.\-_\s](RETRO|xvid|x264|720p|1080p|bluray|brrip|dvdrip|web|webrip|dts|aac|ac3)',
                '',
                source_name,
                flags=re.IGNORECASE
            )

            # First check Videos directory for movie/TV content
            videos_dir = base_dest / "Videos"
            if videos_dir.exists():
                # Try TV show matching first (S##E## pattern)
                show_match = re.match(r'^(.*?)[\.\-_\s]s\d+', clean_source, re.IGNORECASE)
                if show_match:
                    show_name = show_match.group(1)
                    if show_name == 'downloads':  # Skip generic names
                        return None
                        
                    logger.debug(f"Detected TV show: {show_name}")
                    
                    # Find matching show folders
                    matches = []
                    for folder in videos_dir.iterdir():
                        if folder.is_dir() and folder.name.lower() not in generic_dirs:
                            folder_name = folder.name.lower()
                            clean_folder = re.sub(
                                r'[\.\-_\s](RETRO|xvid|x264|720p|1080p|bluray|brrip|dvdrip|web|webrip|dts|aac|ac3)',
                                '',
                                folder_name,
                                flags=re.IGNORECASE
                            )
                            
                            # Check if it's the same show
                            if clean_folder.startswith(show_name):
                                similarity = self._calculate_name_similarity(clean_source, clean_folder)
                                if similarity > 0.8:  # High similarity threshold
                                    matches.append((similarity, folder))
                                    logger.debug(f"Found potential TV match: {folder} (score: {similarity})")
                    
                    # Use best match if found
                    if matches:
                        best_match = max(matches, key=lambda x: x[0])
                        logger.debug(f"Using TV show folder: {best_match[1]}")
                        return best_match[1]

                # Try movie matching
                matches = []
                for folder in videos_dir.iterdir():
                    if folder.is_dir() and folder.name.lower() not in generic_dirs:
                        folder_name = folder.name.lower()
                        clean_folder = re.sub(
                            r'[\.\-_\s](RETRO|xvid|x264|720p|1080p|bluray|brrip|dvdrip|web|webrip|dts|aac|ac3)',
                            '',
                            folder_name,
                            flags=re.IGNORECASE
                        )
                        
                        similarity = self._calculate_name_similarity(clean_source, clean_folder)
                        if similarity > 0.8:
                            matches.append((similarity, folder))
                            logger.debug(f"Found potential movie match: {folder} (score: {similarity})")
                
                if matches:
                    best_match = max(matches, key=lambda x: x[0])
                    logger.debug(f"Using movie folder: {best_match[1]}")
                    return best_match[1]

            return None

        except Exception as e:
            logger.error(f"Error finding media home for {file_path}: {e}")
            return None

    def process_file(self, file_path: Path) -> None:
        try:
            if file_path.suffix.lower() in {'.mkv', '.mp4', '.avi', '.mov'}:
                # Get clean folder structure
                folder_path = self.clean_media_name(file_path)
                dest_dir = Path("/mnt/z/sort/Videos") / folder_path
                
                # Keep original filename
                dest_path = dest_dir / file_path.name
                
                if not self.dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    file_path.rename(dest_path)
                    logger.info(f"Moving to: {dest_path}")
                    
                    # Move related files (.srt, .sub, .idx, etc)
                    related_files = list(file_path.parent.glob(f"{file_path.stem}.*"))
                    for related in related_files:
                        if related != file_path:
                            related_dest = dest_dir / related.name
                            related.rename(related_dest)
                            logger.debug(f"Moving related file: {related.name}")
                else:
                    logger.info(f"Would move: {file_path.name} → {dest_path}")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")

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
        """Extract metadata from AI-generated audio files"""
        try:
            # Get full metadata including format and streams
            cmd = [
                'ffprobe', 
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format', 
                '-show_entries', 
                'format_tags:stream_tags=title,artist,album,comment,lyrics,description',
                str(file_path)
            ]
            
            # Run ffprobe
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            try:
                metadata = json.loads(result.stdout)
            except json.JSONDecodeError:
                # Try tag extraction using other tools
                try:
                    from mutagen import File
                    audio = File(str(file_path))
                    if audio:
                        metadata = {'format': {'tags': audio.tags if audio.tags else {}}}
                    else:
                        return {'is_ai': False, 'tool': ''}
                except ImportError:
                    return {'is_ai': False, 'tool': ''}

            # Extract all available tags
            all_tags = {}
            if 'format' in metadata:
                all_tags.update(metadata['format'].get('tags', {}))
            for stream in metadata.get('streams', []):
                all_tags.update(stream.get('tags', {}))

            # Also check filename for AI patterns since metadata might be missing
            filename = file_path.name.lower()
            name_match = False
            ai_model = None

            # Check filename for AI model names
            if any(x in filename for x in ['claude', 'claude3.5', 'sonnet', 'haiku']):
                name_match = True
                ai_model = 'CLAUDE'
            elif any(x in filename for x in ['grok', 'grok2', 'grok-2']):
                name_match = True
                ai_model = 'GROK'  
            elif any(x in filename for x in ['sonar-huge', 'sonar-large', 'sonar huge', 'sonar large']):
                name_match = True
                ai_model = 'SONAR'
            elif any(x in filename for x in ['ppx', 'ppx-03', 'ppx mini']):
                name_match = True
                ai_model = 'PPX'
            elif any(x in filename for x in ['deepseek', 'deepseek-r1']):
                name_match = True
                ai_model = 'DEEPSEEK'

            if name_match:
                logger.debug(f"AI model detected from filename: {ai_model}")
                return {'is_ai': True, 'tool': ai_model}

            # If no filename match, check metadata
            for field, value in all_tags.items():
                value = str(value).lower()
                
                # Check for AI model names in metadata
                if any(x in value for x in ['claude', 'claude3.5', 'sonnet', 'haiku']):
                    return {'is_ai': True, 'tool': 'CLAUDE'}
                elif any(x in value for x in ['grok', 'grok2', 'grok-2']):
                    return {'is_ai': True, 'tool': 'GROK'}
                elif any(x in value for x in ['sonar-huge', 'sonar-large']):
                    return {'is_ai': True, 'tool': 'SONAR'}
                elif any(x in value for x in ['ppx', 'ppx-03', 'ppx mini']):
                    return {'is_ai': True, 'tool': 'PPX'}
                elif any(x in value for x in ['deepseek', 'deepseek-r1']):
                    return {'is_ai': True, 'tool': 'DEEPSEEK'}

            return {'is_ai': False, 'tool': ''}

        except Exception as e:
            logger.error(f"Error extracting AI audio metadata from {file_path.name}: {e}")
            return {'is_ai': False, 'tool': ''}

    def _check_filename_patterns(self, file_path: Path) -> Dict[str, str]:
        """Check filename for AI generation patterns"""
        name = file_path.stem.lower()
        
        # Known AI audio tool patterns in filenames
        patterns = {
            'suno': [r'suno[-_]', r'bark[-_]', r'audio[-_]craft'],
            'musicgen': [r'mg[-_]', r'musicgen[-_]', r'mubert[-_]'],
            'stable_audio': [r'sa[-_]', r'stable[-_]audio'],
            'riffusion': [r'riff[-_]', r'riffusion[-_]']
        }
        
        # Check for direct tool name matches
        for tool, tool_patterns in patterns.items():
            if any(re.search(pattern, name) for pattern in tool_patterns):
                return {'is_ai': True, 'tool': tool}
        
        # Check for numeric patterns suggesting AI generation
        if re.search(r'[-_](v\d+|gen\d+|iter\d+)[-_]', name):
            return {'is_ai': True, 'tool': 'unknown_ai'}
            
        return {'is_ai': False, 'tool': ''}

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

    def process_directory(self, dir_path: Path, merge: bool = False, recursive: bool = True) -> None:
        """Recursively scan all subdirectories and merge media files to root organization"""
        try:
            if not dir_path.exists():
                logger.error(f"Directory does not exist: {dir_path}")
                return

            self.source_dir = dir_path
            media_files = {
                'videos': [],
                'family': [],
                'tv': []
            }

            # Recursively find all media files
            for item in dir_path.rglob('*'):
                if not item.is_file():
                    continue

                # Skip already organized files
                if self.is_already_organized(item):
                    logger.debug(f"Skipping already organized file: {item}")
                    self.stats['skipped'] += 1
                    continue

                # Check for media files
                if item.suffix.lower() in {'.mkv', '.mp4', '.avi', '.mov'}:
                    self.stats['total'] += 1
                    
                    # Check if it's a TV show episode
                    if re.search(r's\d{2}e\d{2}', item.stem, re.IGNORECASE):
                        media_files['tv'].append(item)
                    else:
                        # Check if it's family content
                        clean_name = re.sub(r'[\.\-_].*$', '', item.stem)  # Remove release info
                        is_family, _ = self.is_family_content(clean_name)
                        if is_family:
                            media_files['family'].append(item)
                        else:
                            media_files['videos'].append(item)

            # Process media files
            for media_type, files in media_files.items():
                self.process_media_group(files, media_type)

            # Save cache after processing
            self._save_cache()

        except Exception as e:
            logger.error(f"Error processing directory {dir_path}: {e}")

    def process_media_group(self, files: List[Path], media_type: str) -> None:
        try:
            if not files:
                return
            
            # Filter out already organized files
            files = [f for f in files if not self.is_already_organized(f)]
            if not files:
                return
            
            base_dest = Path("/mnt/z/sort")
            
            # Group related files by their base names
            media_groups = defaultdict(list)
            
            # First, identify main media files and their clean names
            main_media_files = {}  # base_name -> clean_folder_path
            for file_path in files:
                if file_path.suffix.lower() in {'.mkv', '.mp4', '.avi', '.mov'}:
                    base_name = file_path.stem
                    clean_folder = self.clean_media_name(file_path)
                    main_media_files[base_name] = clean_folder
                    media_groups[base_name].append(file_path)

            # Then, associate related files with their main media files
            for file_path in files:
                if file_path.suffix.lower() not in {'.mkv', '.mp4', '.avi', '.mov'}:
                    base_name = file_path.stem
                    if base_name in main_media_files:
                        media_groups[base_name].append(file_path)
                    else:
                        for media_base in main_media_files:
                            if base_name.startswith(media_base) or media_base.startswith(base_name):
                                media_groups[media_base].append(file_path)
                                break

            # Process each media group
            for base_name, group_files in media_groups.items():
                if base_name in main_media_files:
                    clean_folder = main_media_files[base_name]
                    dest_dir = base_dest / "Videos" / clean_folder

                    if not self.dry_run:
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        for file_path in group_files:
                            dest_path = dest_dir / file_path.name
                            file_path.rename(dest_path)
                            logger.info(f"Moving to: {dest_path}")
                            self.moved_files.append((str(file_path), str(dest_path)))
                    else:
                        for file_path in group_files:
                            logger.info(f"Would move: {file_path.name} → {dest_dir / file_path.name}")
                else:
                    for file_path in group_files:
                        if media_type == 'videos':
                            dest_dir = base_dest / "Videos" / "Uncategorized" / "Unknown"
                        else:
                            dest_dir = base_dest / media_type.capitalize()

                        if not self.dry_run:
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            dest_path = dest_dir / file_path.name
                            file_path.rename(dest_path)
                            logger.info(f"Moving to: {dest_path}")
                            self.moved_files.append((str(file_path), str(dest_path)))
                        else:
                            logger.info(f"Would move: {file_path.name} → {dest_dir / file_path.name}")

        except Exception as e:
            logger.error(f"Error processing media group: {e}")
            logger.debug(f"Failed files: {[f.name for f in files]}")

    def verify_media_folder_moved(self, source_dir: Path, dest_base: Path) -> bool:
        """Verify if media folder content was moved to destination"""
        try:
            # Clean source folder name for matching
            clean_source = re.sub(
                r'[\.\-_\s](RETRO|xvid|x264|720p|1080p|bluray|brrip|dvdrip|web|webrip|dts|aac|ac3)',
                '',
                source_dir.name.lower(),
                flags=re.IGNORECASE
            )

            # Check Videos directory for matching folder
            videos_dir = dest_base / "Videos"
            if videos_dir.exists():
                for folder in videos_dir.iterdir():
                    if folder.is_dir():
                        clean_folder = re.sub(
                            r'[\.\-_\s](RETRO|xvid|x264|720p|1080p|bluray|brrip|dvdrip|web|webrip|dts|aac|ac3)',
                            '',
                            folder.name.lower(),
                            flags=re.IGNORECASE
                        )
                        if clean_source == clean_folder or self._calculate_name_similarity(clean_source, clean_folder) > 0.8:
                            logger.debug(f"Found matching destination: {folder}")
                            return True
            return False
        except Exception as e:
            logger.error(f"Error verifying media folder: {e}")
            return False

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names using Levenshtein distance"""
        try:
            from Levenshtein import ratio
            return ratio(name1.lower(), name2.lower())
        except ImportError:
            # Fallback to simple matching if python-Levenshtein is not installed
            name1 = name1.lower()
            name2 = name2.lower()
            
            # Remove common words and separators
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
            separators = {'.', '-', '_', ' '}
            
            for word in common_words:
                name1 = name1.replace(word, '')
                name2 = name2.replace(word, '')
                
            for sep in separators:
                name1 = name1.replace(sep, '')
                name2 = name2.replace(sep, '')
            
            # Calculate simple similarity ratio
            shorter = min(len(name1), len(name2))
            longer = max(len(name1), len(name2))
            
            if longer == 0:
                return 0.0
                
            matches = sum(1 for i in range(shorter) if name1[i] == name2[i])
            return matches / longer

    def _fix_year(self, year: str, folder_name: str, file_name: str = None) -> str:
        """Fix incorrect years by checking folder name first, then filename"""
        try:
            year_int = int(year)
            current_year = datetime.now().year
            
            # First check folder name for year
            folder_year_match = re.search(r'(?:19|20)(\d{2})', folder_name)
            if folder_year_match:
                folder_year = folder_year_match.group()
                if folder_year != year:
                    logger.debug(f"Found different year in folder name: {folder_year} vs {year}")
                    return folder_year
            
            # Then check filename if provided
            if file_name:
                file_year_match = re.search(r'(?:19|20)(\d{2})', file_name)
                if file_year_match:
                    file_year = file_year_match.group()
                    if file_year != year and file_year != folder_year_match.group() if folder_year_match else None:
                        logger.debug(f"Found different year in filename: {file_year} vs {year}")
                        return file_year

            # If year is in the future, fix it
            if year_int > current_year:
                # Check if switching between centuries fixes it
                if year_int > 2100:  # Likely should be 19xx
                    corrected = f"19{year[-2:]}"
                elif year_int >= 2025:  # Current reasonable future cutoff
                    corrected = f"19{year[-2:]}"
                else:
                    return year  # Keep years between now and 2025
                
                logger.debug(f"Corrected future year {year} to {corrected}")
                return corrected
                
            return year

        except ValueError:
            logger.error(f"Error fixing year: {year}")
            return year

    def clean_media_name(self, file_path: Path) -> str:
        try:
            name = file_path.stem
            parent_folder = file_path.parent.name
            
            # Extract title without year
            title_match = re.match(r'^(.+?)(?:[\.\-_ ](?:19|20)\d{2})?', name, re.IGNORECASE)
            if title_match:
                title = title_match.group(1)
                # Clean up the title
                title = (title
                        .replace('.', ' ')
                        .replace('_', ' ')
                        .strip()
                        .title())
                
                # Check if it's family content and get correct year from IMDb
                is_family, imdb_year = self.is_family_content(title)
                base_folder = "Family" if is_family else "Uncategorized"
                
                if imdb_year:
                    return f"{base_folder}/{title} ({imdb_year})"
                else:
                    # Fallback to year in filename if IMDb lookup fails
                    year_match = re.search(r'(?:19|20)(\d{2})', name)
                    year = year_match.group() if year_match else "Unknown"
                    return f"{base_folder}/{title} ({year})"

            return f"Uncategorized/{name}"

        except Exception as e:
            logger.error(f"Error cleaning media name: {e}")
            return "Uncategorized/Unknown"

    def sanitize_filename(self, name: str) -> str:
        """Make filename POSIX-compatible"""
        # Replace spaces and problematic chars
        name = re.sub(r'[^\w\-\.]', '_', name)
        name = re.sub(r'[\s\.]', '_', name)
        name = re.sub(r'_{2,}', '_', name)  # Collapse multiple underscores
        name = name.strip('_')  # Trim underscores
        return name

    def is_family_content(self, title: str) -> tuple[bool, Optional[str]]:
        """Determine if content is family-friendly using IMDb API and cached results"""
        try:
            # Check cache first
            cache_key = title.lower().strip()
            if cache_key in self.imdb_cache:
                cached = self.imdb_cache[cache_key]
                logger.debug(f"IMDb cache hit: {title}")
                return cached['is_family'], cached['year']

            # Clean up title for better IMDb matching
            clean_title = re.sub(r'[\.\-_\(\)]', ' ', title).strip()
            clean_title = re.sub(r'\s*(?:19|20)\d{2}\s*$', '', clean_title)
            
            # Use IMDb API
            ia = IMDb()
            results = ia.search_movie(clean_title)
            
            if results:
                movie = results[0]
                ia.update(movie)
                
                # Get IMDb year
                imdb_year = str(movie.get('year', ''))
                
                # Check primary genres
                genres = [g.lower() for g in movie.get('genres', [])][:2]
                is_family = False

                if 'animation' in genres or 'family' in genres:
                    logger.debug(f"Family content detected - {title} - Genres: {genres}")
                    is_family = True
                
                # Check ratings
                certificates = movie.get('certificates', [])
                family_ratings = {'USA:G', 'USA:PG', 'UK:U', 'UK:PG'}
                adult_ratings = {'R', 'NC-17', '18', 'MA15+', 'TV-MA'}
                
                has_family_rating = any(cert in family_ratings for cert in certificates)
                has_adult_rating = any(rating in cert for cert in certificates for rating in adult_ratings)
                
                if has_family_rating and not has_adult_rating:
                    logger.debug(f"Family content detected - {title} - Ratings: {certificates}")
                    is_family = True

                # Cache the result
                self.imdb_cache[cache_key] = {
                    'is_family': is_family,
                    'year': imdb_year,
                    'genres': genres,
                    'certificates': certificates,
                    'cached_date': datetime.now().isoformat()
                }
                self._save_imdb_cache()

                return is_family, imdb_year

            return False, None

        except Exception as e:
            logger.error(f"Error checking family content for {title}: {e}")
            return False, None

    def audit_family_content(self, base_dir: Path = None) -> dict:
        """Audit and fix movies in Videos directories"""
        try:
            current_year = datetime.now().year
            audit_results = {
                'year_corrections': [],
                'category_moves': [],
                'total_processed': 0,
                'empty_dirs_removed': 0
            }

            # Check both Family and Movies directories
            family_dir = Path(base_dir) / "Family"
            movies_dir = Path(base_dir) / "Movies"

            logger.info(f"Auditing content categories and years...")
            
            # Process Family directory first
            if family_dir.exists():
                for item in family_dir.glob("**/*"):
                    if not item.is_file() or item.suffix.lower() not in {'.mkv', '.mp4', '.avi', '.mov'}:
                        continue

                    audit_results['total_processed'] += 1
                    content_name = item.parent.name
                    clean_name = re.sub(r'\s*\(\d{4}\)$', '', content_name)
                    
                    # Check if it's actually family content
                    is_family, imdb_year = self.is_family_content(clean_name)
                    
                    if imdb_year:
                        # Fix incorrect years
                        year_match = re.search(r'\((\d{4})\)$', content_name)
                        if year_match and year_match.group(1) != imdb_year:
                            new_name = f"{clean_name} ({imdb_year})"
                            audit_results['year_corrections'].append(
                                f"{content_name} → {new_name}"
                            )
                            if not self.dry_run:
                                new_dir = (movies_dir if not is_family else family_dir) / new_name
                                new_dir.mkdir(parents=True, exist_ok=True)
                                item.rename(new_dir / item.name)
                        
                        # Move non-family content to Movies
                        if not is_family:
                            dest_dir = movies_dir / content_name
                            audit_results['category_moves'].append(
                                f"Family/{content_name} → Movies"
                            )
                            if not self.dry_run:
                                dest_dir.mkdir(parents=True, exist_ok=True)
                                item.rename(dest_dir / item.name)

            # Clean up empty directories
            if not self.dry_run:
                for dir_path in [family_dir, movies_dir]:
                    for path in sorted(dir_path.glob("**/*"), key=lambda x: len(str(x)), reverse=True):
                        if path.is_dir() and not any(path.iterdir()):
                            path.rmdir()
                            audit_results['empty_dirs_removed'] += 1

            return audit_results

        except Exception as e:
            logger.error(f"Error during family content audit: {e}")
            return {}

    def is_already_organized(self, file_path: Path) -> bool:
        """Check if file is already organized using smart caching"""
        try:
            if str(self.dest_base) in str(file_path):
                # Get file metadata for cache key
                stat = file_path.stat()
                cache_key = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
                
                # Check cache with metadata
                if cache_key in self.processed_cache:
                    cache_entry = self.processed_cache[cache_key]
                    # Verify nothing has changed
                    if (cache_entry['size'] == stat.st_size and 
                        cache_entry['mtime'] == stat.st_mtime):
                        return True
                
                # Add to cache if in organized structure
                category = next((p for p in file_path.parts 
                               if p in {'Videos', 'Music', 'Documents', 'Images'}), None)
                if category:
                    self.processed_cache[cache_key] = {
                        'path': str(file_path),
                        'category': category,
                        'size': stat.st_size,
                        'mtime': stat.st_mtime,
                        'processed_date': datetime.now().isoformat()
                    }
                    self.cache_modified = True  # Set modified flag
                    return True
                    
            return False

        except Exception as e:
            logger.error(f"Error checking if file is organized: {e}")
            return False

    def _is_ai_generated_image(self, file_path: Path) -> bool:
        """Detect if an image is AI-generated by checking metadata patterns"""
        try:
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS
            except ImportError:
                logger.warning("PIL not installed. Skipping AI image detection.")
                return False

            if file_path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
                return False

            with Image.open(file_path) as img:
                # Check common AI image metadata patterns
                metadata = {
                    TAGS.get(k, k): v 
                    for k, v in img.getexif().items()
                }
                
                # Known AI metadata patterns
                ai_indicators = {
                    'Software': [
                        'Stable Diffusion', 'DALL-E', 'Midjourney', 
                        'RunwayML', 'disco_diffusion', 'deepdream'
                    ],
                    'Artist': [
                        'AI', 'Artificial Intelligence', 'Generated',
                        'Stable Diffusion', 'DALL-E', 'Midjourney'
                    ],
                    'Generator': ['AI', 'ML', 'GAN', 'Diffusion'],
                    'Comment': [
                        'prompt:', 'seed:', 'cfg:', 'steps:',
                        'sampler:', 'negative prompt:'
                    ]
                }

                # Check image metadata
                for field, patterns in ai_indicators.items():
                    if field in metadata:
                        value = str(metadata[field]).lower()
                        if any(pattern.lower() in value for pattern in patterns):
                            logger.debug(f"AI image detected via {field}: {value}")
                            return True

                # Check PNG metadata (common for SD and Midjourney)
                if hasattr(img, 'text'):
                    png_text = {k.lower(): v.lower() for k, v in img.text.items()}
                    
                    # Check for common AI-related parameters
                    ai_params = {
                        'prompt', 'negative_prompt', 'seed', 'steps', 
                        'cfg_scale', 'sampler', 'model', 'scheduler'
                    }
                    if any(param in png_text for param in ai_params):
                        logger.debug(f"AI image detected via PNG metadata")
                        return True

                # Check dimensions (common AI resolutions)
                width, height = img.size
                ai_resolutions = {
                    (512, 512), (768, 768), (1024, 1024),  # Square formats
                    (512, 768), (768, 512),                 # Portrait/Landscape SD
                    (1024, 1536), (1536, 1024),            # Portrait/Landscape MJ
                    (1024, 576), (576, 1024)               # Widescreen
                }
                if (width, height) in ai_resolutions:
                    logger.debug(f"Possible AI image detected via dimensions: {width}x{height}")
                    # Don't return True here as these dimensions alone aren't conclusive

            return False

        except Exception as e:
            logger.error(f"Error checking AI image metadata: {e}")
            return False

    def save_execution_log(self, results: dict) -> None:
        """Save execution results to JSON log file"""
        log_dir = Path.home() / '.odd' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats,
            'moved_files': self.moved_files,
            'audit_results': results,
            'source_dir': str(self.source_dir)
        }
        
        # Save to daily log file
        date_str = datetime.now().strftime('%Y-%m-%d')
        log_file = log_dir / f'odd_log_{date_str}.json'
        
        try:
            if log_file.exists():
                with open(log_file) as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving log: {e}")

    def get_recent_history(self, days: int = 7) -> dict:
        """Get organization history from logs"""
        log_dir = Path.home() / '.odd' / 'logs'
        history = {
            'total_processed': 0,
            'family_moves': 0,
            'year_corrections': 0,
            'recent_additions': [],
            'by_date': {}
        }
        
        if not log_dir.exists():
            return history
            
        # Get logs from last N days
        cutoff = datetime.now() - timedelta(days=days)
        for log_file in log_dir.glob('odd_log_*.json'):
            try:
                with open(log_file) as f:
                    daily_logs = json.load(f)
                    
                for entry in daily_logs:
                    entry_date = datetime.fromisoformat(entry['timestamp'])
                    if entry_date >= cutoff:
                        date_str = entry_date.strftime('%Y-%m-%d')
                        if date_str not in history['by_date']:
                            history['by_date'][date_str] = {
                                'processed': 0,
                                'family_moves': 0,
                                'year_fixes': 0
                            }
                        
                        # Update stats
                        history['total_processed'] += entry['stats']['total']
                        history['by_date'][date_str]['processed'] += entry['stats']['total']
                        
                        if 'audit_results' in entry:
                            history['family_moves'] += len(entry['audit_results'].get('category_moves', []))
                            history['year_corrections'] += len(entry['audit_results'].get('year_corrections', []))
                            history['by_date'][date_str]['family_moves'] += len(entry['audit_results'].get('category_moves', []))
                            history['by_date'][date_str]['year_fixes'] += len(entry['audit_results'].get('year_corrections', []))
                        
                        # Track recent family additions
                        for moved in entry.get('moved_files', []):
                            if 'Family' in str(moved[1]):
                                history['recent_additions'].append({
                                    'name': Path(moved[1]).parent.name,
                                    'date': entry['timestamp']
                                })
                                
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")
        
        return history

    def _print_audit_summary(self, audit_results: dict) -> None:
        """Print comprehensive summary of organization results"""
        console = Console()
        console.print("\n=== Organization Summary ===", style="bold blue")
        
        if self.dry_run:
            console.print("DRY RUN - No files were actually moved", style="bold red")
        
        # Directory Statistics
        videos_dir = Path("/mnt/z/sort/Videos")
        if videos_dir.exists():
            family_dir = videos_dir / "Family"
            movies_dir = videos_dir / "Movies"
            tv_dir = videos_dir / "TV"
            other_dir = videos_dir / "Other"  # For non-IMDB video content
            
            console.print("\nDirectory Status:", style="bold yellow")
            
            # Show counts for each category
            for dir_path, label in [
                (family_dir, "Family Movies"),
                (movies_dir, "Movies"),
                (tv_dir, "TV Shows"),
                (other_dir, "Other Videos")  # Non-IMDB video content
            ]:
                if dir_path.exists():
                    count = len(list(dir_path.glob("*/")))
                    console.print(f"{label}: {count}")

            # Show media found in subdirectories
            console.print("\nMedia Found in Subdirectories:", style="cyan")
            if self.stats.get('subdir_media', 0) > 0:
                console.print(f"Total media files found: {self.stats['subdir_media']}")
                console.print(f"• Movies to be merged: {self.stats.get('movies_to_merge', 0)}")
                console.print(f"• TV episodes to be merged: {self.stats.get('tv_to_merge', 0)}")
                console.print(f"• Family content to be merged: {self.stats.get('family_to_merge', 0)}")
                console.print(f"• Other videos to be merged: {self.stats.get('other_to_merge', 0)}")

            # Show recent changes
            recent = [d for d in family_dir.glob("*/") 
                     if d.stat().st_mtime > time.time() - 86400]
            if recent:
                console.print("\nRecent Changes (Last 24h):", style="green")
                for d in sorted(recent, key=lambda x: x.stat().st_mtime, reverse=True):
                    console.print(f"  • {d.name}")

        # Show processing statistics
        console.print("\nProcessing Results:", style="bold yellow")
        console.print(f"Total Files Processed: {self.stats['total']}")
        console.print(f"Successfully Moved: {self.stats['success']}")
        console.print(f"Merged from Subdirectories: {self.stats.get('merged', 0)}")
        console.print(f"Errors: {self.stats['errors']}")
        console.print(f"Duplicates Found: {self.stats['duplicates']}")
        console.print(f"Skipped (Already Organized): {self.stats['skipped']}")

        console.print("\n===============================", style="bold blue")

    def audit_organization(self, base_dir: Path = None) -> dict:
        """Audit and fix organization issues"""
        try:
            audit_results = {
                'category_moves': [],
                'year_corrections': [],
                'misplaced_files': [],
                'total_processed': 0,
                'empty_dirs_removed': 0
            }

            base_dir = base_dir or self.dest_base
            logger.info(f"Auditing organization structure in {base_dir}...")

            # Define correct locations for file types
            file_categories = {
                'Images': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ico'},
                'Videos': {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.webm', '.flv'},
                'Documents': {'.pdf', '.doc', '.docx', '.txt', '.xlsx', '.csv', '.rtf'},
                'Music': {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac'},
                'Code': {'.py', '.js', '.html', '.css', '.java', '.cpp', '.h', '.php'},
            }

            # Define paths to exclude
            excluded_paths = {
                'venv', 'site-packages', '.git', 'node_modules', '__pycache__',
                'build', 'dist', 'PycharmProjects', 'Scripts'
            }

            def should_exclude(path: Path) -> bool:
                return any(exclude in path.parts for exclude in excluded_paths)

            # Process each file in directory tree
            for root, _, files in os.walk(base_dir):
                root_path = Path(root)
                
                # Skip excluded directories
                if should_exclude(root_path):
                    continue

                for file in files:
                    file_path = root_path / file
                    ext = file_path.suffix.lower()
                    audit_results['total_processed'] += 1

                    # Skip excluded paths and special directories
                    if should_exclude(file_path):
                        continue

                    # Find correct category for file
                    correct_category = None
                    for category, extensions in file_categories.items():
                        if ext in extensions:
                            correct_category = category
                            break

                    if correct_category:
                        current_category = next((p for p in file_path.parts 
                                              if p in file_categories.keys()), None)
                        
                        # If file is in wrong category or no category
                        if current_category != correct_category:
                            dest_dir = base_dir / correct_category
                            dest_path = dest_dir / file_path.name

                            # Handle duplicates
                            counter = 1
                            while dest_path.exists():
                                stem = dest_path.stem
                                if ' (copy ' in stem:
                                    stem = stem[:stem.rfind(' (copy ')]
                                dest_path = dest_dir / f"{stem} (copy {counter}){dest_path.suffix}"
                                counter += 1

                            audit_results['misplaced_files'].append(
                                f"{file_path} → {dest_path}"
                            )

                            if not self.dry_run:
                                dest_dir.mkdir(parents=True, exist_ok=True)
                                file_path.rename(dest_path)
                                logger.info(f"Moving misplaced file: {file_path} → {dest_path}")

            # Run family content audit for videos
            videos_dir = base_dir / "Videos"
            if videos_dir.exists():
                family_audit = self.audit_family_content(videos_dir)
                audit_results['year_corrections'].extend(family_audit.get('year_corrections', []))
                audit_results['category_moves'].extend(family_audit.get('category_moves', []))

            return audit_results

        except Exception as e:
            logger.error(f"Error during organization audit: {e}")
            return {}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Organize files in directories.')
    parser.add_argument('-s', '--source', type=str, help='Source directory')
    parser.add_argument('-c', '--config', type=str, help='Config file path')
    parser.add_argument('-d', '--dry-run', action='store_true', help='Dry run')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-r', '--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('-m', '--merge', action='store_true', help='Merge subdirectories')
    parser.add_argument('--summary', action='store_true', help='Show detailed summary')
    parser.add_argument('--report', choices=['text', 'json', 'csv', 'html'], default='text')
    parser.add_argument('--report-dir', type=str, help='Custom report directory')
    parser.add_argument('--ai-only', action='store_true', help='Process only AI-generated files')
    parser.add_argument('--audit', action='store_true', 
                       help='Audit and fix organization issues (misplaced files, incorrect categories, years)')
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

    # Track results for logging
    results = {
        'audit_results': {},
        'processed_time': datetime.now().isoformat(),
        'command_args': vars(args)
    }

    if args.audit:  # Changed from args.audit_family to args.audit
        logger.info("Running organization audit...")
        audit_results = organizer.audit_organization(Path("/mnt/z/sort"))
        results['audit_results'] = audit_results

    organizer.process_directory(source_dir, args.merge, args.recursive)
    
    # Update results with final stats
    results.update({
        'final_stats': organizer.stats,
        'source_dir': str(source_dir),
        'dry_run': args.dry_run
    })

    # Save execution log
    organizer.save_execution_log(results)
    
    # Always show summary report
    console = Console()
    console.print("\n=== Organization Summary ===", style="bold blue")
    
    if args.dry_run:
        console.print("DRY RUN - No files were actually moved", style="bold red")
    
    # Directory Statistics
    videos_dir = Path("/mnt/z/sort/Videos")
    if videos_dir.exists():
        family_dir = videos_dir / "Family"
        uncat_dir = videos_dir / "Uncategorized"
        
        console.print("\nCurrent Directory Status:", style="bold yellow")
        if family_dir.exists():
            family_count = len(list(family_dir.glob("*/")))
            console.print(f"Family Movies: {family_count}")
            # Show recent additions (last 24h)
            recent = [d for d in family_dir.glob("*/") 
                     if d.stat().st_mtime > time.time() - 86400]
            if recent:
                console.print("\nRecent Family Additions:", style="green")
                for d in sorted(recent, key=lambda x: x.stat().st_mtime, reverse=True):
                    console.print(f"  • {d.name}")
        
        if uncat_dir.exists():
            uncat_count = len(list(uncat_dir.glob("*/")))
            console.print(f"Uncategorized Movies: {uncat_count}")
    
    # Processing Results
    console.print("\nProcessing Statistics:", style="bold yellow")
    console.print(f"Total Files Processed: {organizer.stats['total']}")
    console.print(f"Successfully Moved: {organizer.stats['success']}")
    console.print(f"Errors: {organizer.stats['errors']}")
    console.print(f"Duplicates Found: {organizer.stats['duplicates']}")
    console.print(f"Skipped (Already Organized): {organizer.stats['skipped']}")

    if args.audit and results['audit_results']:  # Changed from args.audit_family to args.audit
        console.print("\nAudit Results:", style="bold yellow")
        if results['audit_results'].get('year_corrections'):
            console.print("Year Corrections:", style="red")
            for correction in results['audit_results']['year_corrections']:
                console.print(f"  • {correction}")
        if results['audit_results'].get('category_moves'):
            console.print("Category Moves:", style="green")
            for move in results['audit_results']['category_moves']:
                console.print(f"  • {move}")
        if results['audit_results'].get('misplaced_files'):
            console.print("Misplaced Files:", style="yellow")
            for move in results['audit_results']['misplaced_files']:
                console.print(f"  • {move}")

    console.print("\n===============================", style="bold blue")

if __name__ == "__main__":
    main()