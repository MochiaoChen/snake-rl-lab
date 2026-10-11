"""Read a saved training progress snapshot."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--run',default='runs/v2-global')
a=p.parse_args()
path=Path(a.run)/'progress.json'
print(json.dumps(json.loads(path.read_text()),indent=2,ensure_ascii=False) if path.exists()
      else 'Training has not reached the first checkpoint yet.')
