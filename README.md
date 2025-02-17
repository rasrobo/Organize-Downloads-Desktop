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


