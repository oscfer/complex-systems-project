"""Rebuild presentation from a completed scale bundle, without running science.

Usage: python tools/rebuild_scale_animations.py /path/to/final_scale_bundle
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.visualization.scale_animation import rebuild_animations

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path)
    args=parser.parse_args()
    result=rebuild_animations(args.bundle)
    print('Animation index:',args.bundle/'analysis/animations/index.html')
    print('Saved comparisons:',sum(p['comparisons'] for p in result['players'].values()))
