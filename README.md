# 🗂️ ODD (Organize Downloads Desktop)

Intelligent file organization with AI content detection, specializing in managing AI-generated content from popular creative tools.

## ✨ Features

### 🤖 AI Content Detection & Organization
- **Audio**
  - Suno/AI Test Kitchen: Smart detection of themes and patterns
  - Udio: Automated classification
  - MusicGen/AudioCraft: Metadata extraction
- **Video**
  - Pika Labs: Motion tracking
  - Runway: Scene detection
  - Gen-1: Style recognition
  - Stable Video: Frame analysis
- **Images**
  - Midjourney: Prompt preservation
  - DALL·E: Metadata extraction
  - Stable Diffusion: Parameter tracking

### 📁 Smart Organization
- POSIX-compliant filename sanitization
- Metadata preservation and enhancement
- Intelligent categorization by AI tool
- Creator attribution maintenance
- Version tracking and iteration management

### 📊 Rich Reporting
- Real-time console output with progress
- Detailed JSON/CSV export options
- Comprehensive movement logs
- Duplicate detection reports

### 🛠️ Audit Functionality
- Fixes misplaced files (e.g., images in Documents folder)
- Corrects movie years using IMDb data
- Properly categorizes family vs adult content
- Moves content to appropriate directories
- Removes empty directories
- Generates detailed report of changes

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/rasrobo/organize-downloads-desktop.git
cd organize-downloads-desktop
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Basic usage
python odd.py -s ~/Downloads -v
```

## 📌 Usage Examples

```bash
# Organize with different options
python odd.py -s ~/Downloads --ai-only          # Process only AI-generated content
python odd.py -s ~/Downloads --dry-run          # Preview changes
python odd.py -s ~/Downloads -v --debug         # Verbose debug output

# Generate reports
python odd.py -s ~/Downloads --report json      # JSON report
python odd.py -s ~/Downloads --report csv       # CSV report

# Basic organization
python odd.py -s /path/to/source -r -m

# Run with audit to fix organization issues
python odd.py -s /path/to/source --audit
```

## ⚙️ Configuration

Create `config.yaml` in the project root:

```yaml
downloads:
  destinations:
    music:
      path: "Music"
      ai_subdir: "AI_Generated"
      tools:
        - suno
        - udio
        - musicgen
    videos:
      path: "Videos"
      ai_subdir: "AI_Generated"
      tools:
        - pika
        - runway
        - gen1
    images:
      path: "Pictures"
      ai_subdir: "AI_Generated"
      tools:
        - midjourney
        - dalle
        - sd
```

## 📁 Directory Structure

```
Downloads/
├── Music/
│   └── AI_Generated/
│       ├── SUNO/
│       │   └── Creator_Name-Track_Title-v1.mp3
│       ├── UDIO/
│       └── MUSICGEN/
├── Videos/
│   └── AI_Generated/
│       ├── PIKA/
│       ├── RUNWAY/
│       └── GEN1/
└── Pictures/
    └── AI_Generated/
        ├── MIDJOURNEY/
        ├── DALLE/
        └── SD/
```

## 🛠️ Requirements

- Python 3.6+
- ffmpeg/ffprobe for media metadata
- Rich for console output
- PyYAML for configuration
- IMDbPY for media detection

## 📝 License

MIT License - See LICENSE file for details

## 👥 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Options

- `-s, --source`: Source directory to organize
- `-r, --recursive`: Process directories recursively
- `-m, --merge`: Merge processed subfolders with parent
- `--audit`: Run organization audit to fix categorization issues
- `-d, --dry-run`: Show what would be done without making changes
- `-v, --verbose`: Show detailed progress

## Donations

If you find this software useful and would like to support its development, you can buy me a coffee! Your support is greatly appreciated.

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/default-orange.png)](https://buymeacoffee.com/robodigitalis)

## Contributing
Contributions welcome! Please feel free to submit a Pull Request.

## License
MIT License

---

Keywords: file organization, media organizer, automatic file sorting, media library manager, file categorization, folder structure optimizer, media metadata extraction, duplicate file detection, AI-generated content detection, family content detection, video organization, music organization, photo organization, document sorting, directory cleanup, automated file management, file classification, IMDb integration, media renaming tool, file deduplication


