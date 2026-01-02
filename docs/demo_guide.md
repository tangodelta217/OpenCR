# OpenCR Demo Guide

## Live Fuel Gauge Demo

The demo module provides a live terminal visualization of the baseline model predictions.

### Quick Start

```bash
# Run demo on a baseline run (streams predictions)
opencr demo run runs/baseline --subject S01 --delay 300

# Run a quick simulated demo
opencr demo fuel-gauge --threshold 0.7 --duration 10 --ascii
```

### Recording a GIF/Video

#### Using Windows Terminal

1. **Recommended Tool: [terminalizer](https://github.com/faressoft/terminalizer)**

   ```bash
   npm install -g terminalizer
   terminalizer record demo --skip-sharing
   # Run your demo command
   terminalizer render demo -o demo.gif
   ```

2. **Alternative: Windows PowerToys Screen Recorder**
   - Press `Win+Shift+R` to start recording
   - Run your demo command
   - Press `Win+Shift+R` to stop

#### Using OBS Studio

1. Open OBS Studio
2. Add "Window Capture" source, select terminal
3. Click "Start Recording"
4. Run `opencr demo run runs/baseline`
5. Click "Stop Recording"

#### Using asciinema (cross-platform)

```bash
pip install asciinema
asciinema rec demo.cast
# Run your demo
# Press Ctrl+D to stop
asciinema upload demo.cast
```

### Demo Commands

| Command | Description |
|---------|-------------|
| `opencr demo run` | Stream predictions from a baseline run |
| `opencr demo fuel-gauge` | Quick simulated demo with fake data |

### Demo Output

The demo displays:

```
+----------------------------------------+
| OpenCR Live Demo                       |
| Subject:    S01                        |
| Sample:     15/30                      |
|                                        |
| True:       2                          |
| Predicted:  2  o                       |
|                                        |
| Accuracy:   80.0%                      |
|                                        |
| FUEL GAUGE                             |
| [##########----------]  80.0%          |
|                                        |
| Trend: Improving                       |
| Status: HIGH CONFIDENCE                |
+----------------------------------------+
```


### Tips for Great Demo Videos

1. **Use a large font size** (at least 16pt)
2. **Use a dark terminal theme**
3. **Set delay to 300-500ms** for visible updates
4. **Record in 1080p or higher**
5. **Keep it under 30 seconds**
