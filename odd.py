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

# Add this after the imports, before the FileOrganizer class
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
            'duplicates': 0
        }
        self.moved_files = []
        self.file_hashes = defaultdict(list)  # SHA-256 hash -> list of file paths

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

    def process_file(self, file_path: Path) -> None:
        """Process a single file"""
        try:
            self.stats['total'] += 1
            file_ext = file_path.suffix.lower()
            
            # Special handling for audio files
            if file_ext in ['.mp3', '.wav', '.ogg', '.m4a']:
                metadata = self.extract_ai_audio_metadata(file_path)
                dest_path = self.process_audio_file(file_path, metadata)
            else:
                # Get downloads config with proper error handling
                downloads_config = self.config.get('downloads', {})
                
                # First check if it's a video file and extract metadata
                if file_ext.lower() in ['.mp4', '.mov', '.avi']:
                    video_metadata = self.get_video_metadata(file_path)
                    ai_metadata = self.extract_ai_video_metadata(file_path)
                    
                    # Check if it's an AI-generated video
                    if ai_metadata.get('tool') or any(p in file_path.name.lower() 
                        for p in downloads_config.get('patterns', {}).get('ai_video', [])):
                        
                        # Generate new filename with metadata
                        new_filename = self.generate_ai_video_filename(file_path, 
                            {**video_metadata, **ai_metadata})
                        
                        # Create AI video destination path
                        dest_path = (file_path.parent / 'Videos' / 'AI_Generated' / 
                            ai_metadata.get('tool', 'Other') / new_filename)
                    else:
                        # Regular video with metadata
                        new_filename = self.generate_descriptive_filename(file_path, video_metadata)
                        dest_path = file_path.parent / 'Videos' / new_filename
                        
                else:
                    # Handle other file types
                    dest_path = None
                    for dest_type, extensions in downloads_config.get('extensions', {}).items():
                        if file_ext in extensions:
                            dest_folder = downloads_config['destinations'][dest_type]
                            dest_path = file_path.parent / dest_folder / file_path.name
                            break

            if dest_path:
                try:
                    # Ensure parent directories exist
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Check if destination exists
                    if dest_path.exists():
                        # Add counter to filename
                        counter = 1
                        while dest_path.exists():
                            new_name = f"{dest_path.stem}_{counter}{dest_path.suffix}"
                            dest_path = dest_path.parent / new_name
                            counter += 1
                    
                    if not self.dry_run:
                        file_path.rename(dest_path)
                        self.stats['success'] += 1
                        self.moved_files.append((str(file_path), str(dest_path)))
                        logger.info(f"Moving: {file_path.name} → {dest_path}")
                    else:
                        logger.info(f"Would move: {file_path.name} → {dest_path}")
                        
                except PermissionError:
                    logger.error(f"Permission denied: {file_path}")
                    self.stats['errors'] += 1
                except Exception as e:
                    logger.error(f"Error moving file {file_path}: {e}")
                    self.stats['errors'] += 1
            else:
                logger.debug(f"No destination found for: {file_path.name}")
                self.stats['skipped'] = self.stats.get('skipped', 0) + 1
                    
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
        """Extract metadata from AI-generated audio files"""
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
                
                # Common audio metadata fields
                info = {
                    'title': format_tags.get('title', ''),
                    'artist': format_tags.get('artist', ''),
                    'album': format_tags.get('album', ''),
                    'genre': format_tags.get('genre', ''),
                    'prompt': format_tags.get('prompt', ''),
                    'tool': '',
                    'is_ai': False
                }
                
                # AI tool detection
                filename = file_path.name.lower()
                if "ai_test_kitchen" in filename or "atk_" in filename:
                    info['tool'] = 'suno'
                    info['is_ai'] = True
                elif "udio_" in filename or "udio-" in filename:
                    info['tool'] = 'udio'
                    info['is_ai'] = True
                elif any(p in filename for p in ['musicgen_', 'audiogen_', 'audiocraft_']):
                    info['tool'] = 'musicgen'
                    info['is_ai'] = True
                
                # Check metadata fields for AI markers
                description = format_tags.get('description', '').lower()
                comment = format_tags.get('comment', '').lower()
                software = format_tags.get('software', '').lower()
                
                ai_markers = ['suno', 'udio', 'ai generated', 'ai test kitchen', 
                             'musicgen', 'audiocraft', 'synthetic']
                
                if any(marker in txt for marker in ai_markers 
                      for txt in [description, comment, software]):
                    info['is_ai'] = True
                    
                    # Try to determine tool if not already set
                    if not info['tool']:
                        if 'suno' in description or 'suno' in comment:
                            info['tool'] = 'suno'
                        elif 'udio' in description or 'udio' in software:
                            info['tool'] = 'udio'
                        elif 'musicgen' in description or 'audiocraft' in comment:
                            info['tool'] = 'musicgen'
                
                return info
                
        except Exception as e:
            logger.debug(f"Error extracting AI audio metadata: {e}")
        return {}

    def process_audio_file(self, file_path: Path, metadata: Dict[str, str]) -> Path:
        """Process and organize audio file based on metadata"""
        if metadata.get('tool') or "AI_Test_Kitchen" in file_path.name:
            # AI-generated audio
            tool = metadata.get('tool', 'suno')  # Default to suno for AI Test Kitchen files
            base_path = file_path.parent / 'Music/AI_Generated' / tool.upper()
        else:
            # Regular audio
            base_path = file_path.parent / 'Music'
        
        # Generate new filename
        new_name = self.sanitize_filename(file_path.name)
        if "AI_Test_Kitchen" in new_name:
            # Special handling for AI Test Kitchen files
            new_name = new_name.replace("AI_Test_Kitchen_", "SUNO_")
            
        return base_path / new_name

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
    
    for file_path in source_dir.glob('*'):
        if file_path.is_file():
            organizer.process_file(file_path)
    
    if args.summary:
        organizer.generate_report(args.report, args.report_dir)

if __name__ == "__main__":
    main()