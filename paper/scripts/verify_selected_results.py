#!/usr/bin/env python3
"""Validate the six selected paper experiments, raw test files, and visualizations."""
import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SELECTED=ROOT/'paper/selected_results'

def digest(stream):
 h=hashlib.sha256()
 for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 selection=json.loads((SELECTED/'selection.json').read_text());manifest=json.loads((SELECTED/'artifact_manifest.json').read_text())
 spec=selection['test_protocol'];split=ROOT/spec['split_manifest']
 with split.open('rb') as f:assert digest(f)==spec['split_sha256']
 maps=json.loads(split.read_text())['test_maps'];reference=None;summaries={r['experiment']:r for r in csv.DictReader((SELECTED/'summary.csv').open())}
 assert len(selection['experiments'])==len(summaries)==6
 for exp in selection['experiments']:
  d=ROOT/exp['data_directory'];meta=json.loads((d/'evaluation_metadata.json').read_text());rows=list(csv.DictReader((d/'episodes.csv').open()))
  assert len(rows)==len({r['map_id'] for r in rows})==1133
  assert [r['map_id'] for r in rows]==maps
  assert all(int(r['eval_seed'])==4777 and not r.get('error') for r in rows)
  assert meta['eval_seeds']==[4777] and meta['evaluation_mode']=='single_global_seed_greedy' and meta['action_selection']=='greedy_argmax'
  assert meta['target_coverage']==.99 and meta['max_steps']==128 and meta['max_path_length'] is None
  assert meta['split_manifest_sha256']==spec['split_sha256'] and meta['checkpoint_sha256']==exp['checkpoint_sha256']
  assert meta['reseed_between_maps'] is False
  starts=[(r['map_id'],r['start_cell_x'],r['start_cell_y']) for r in rows]
  if reference is None:reference=starts
  else:assert starts==reference
  row=summaries[exp['name']];assert sum(r['success'].lower()=='true' for r in rows)==int(row['successes'])
  for field,csvfield in [('mean_path_all','path_length'),('mean_steps','steps'),('mean_coverage','final_coverage')]:assert abs(statistics.mean(float(r[csvfield]) for r in rows)-float(row[field]))<1e-10
  figures=list((ROOT/exp['visualization_directory']).glob('*.png'));assert len(figures)==6
  print('PASS',exp['name'],'1133 maps; seed4777; greedy; paired starts; 6 figures')
 file_checks={}
 for item in manifest['records']:
  assert item['kind'] in ['raw_test_data','test_visualization'] and item['storage']=='file'
  previous=file_checks.setdefault(item['path'],(item['sha256'],item['bytes']))
  assert previous==(item['sha256'],item['bytes'])
 assert len(manifest['records'])==64 and len(file_checks)==64
 assert sum(x['kind']=='raw_test_data' for x in manifest['records'])==28
 assert sum(x['kind']=='test_visualization' for x in manifest['records'])==36
 for name,(expected,size) in file_checks.items():
  p=ROOT/name;assert p.is_file() and p.stat().st_size==size,name
  with p.open('rb') as f:assert digest(f)==expected,name
 assert not manifest.get('archives')
 print('PASS 28 original CSV/JSON files and 36 PNGs; no training records or new logs')

if __name__=='__main__':main()
