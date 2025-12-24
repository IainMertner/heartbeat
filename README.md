# heartbeat

```bash
pyinstaller --onefile --windowed `
 --add-data "scripts;scripts" `
 --add-data "raw;raw" `
 --add-data "output;output" `
 main.py