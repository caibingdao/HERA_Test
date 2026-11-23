"""Refresh smoke-training plots without touching the training process."""
import argparse,csv,json,os,time
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/hera-smoke-plots-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
options=argparse.ArgumentParser()
options.add_argument('--run-dir',type=Path,required=True)
options.add_argument('--total-episodes',type=int,default=10000)
options.add_argument('--streak',type=int,required=True)
options.add_argument('--watch',action='store_true')
ARGS=options.parse_args()
RUN=ARGS.run_dir.resolve()
TOTAL=ARGS.total_episodes
STREAK=ARGS.streak
CONTROL=RUN.with_name(RUN.name+'_control')
OUT=RUN/'plots'

def draw():
    if not (RUN/'episodes.csv').exists():return
    rows=list(csv.DictReader((RUN/'episodes.csv').open()))
    rows=[r for r in rows if r.get('recovery_triggers') is not None]
    if not rows:return
    OUT.mkdir(exist_ok=True)
    xs=[int(r['completed_episodes']) for r in rows]
    def series(key):return [float(r[key]) for r in rows]
    def smooth(values):return [sum(values[max(0,i-19):i+1])/len(values[max(0,i-19):i+1]) for i in range(len(values))]
    fig,axs=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
    fig.suptitle(f'HERA | seed 4777 | {len(rows)}/{TOTAL} episodes | streak {STREAK} | free-cell threshold < 1',fontsize=15)
    coverage=[100*v for v in series('coverage')]
    axs[0,0].plot(xs,coverage,alpha=.3,color='steelblue');axs[0,0].plot(xs,smooth(coverage),color='steelblue',label='20-episode mean');axs[0,0].set(title='Final coverage',ylabel='Coverage (%)',ylim=(0,102));axs[0,0].legend()
    success=[100*(r['success'].lower()=='true') for r in rows]
    axs[0,1].plot(xs,smooth(success),color='seagreen');axs[0,1].set(title='Training success (20-episode mean)',ylabel='Success (%)',ylim=(-2,102))
    axs[0,2].plot(xs,series('episode_steps'),alpha=.35);axs[0,2].plot(xs,smooth(series('episode_steps')));axs[0,2].axhline(128,color='gray',ls='--');axs[0,2].set(title='Episode length',ylabel='Decisions')
    axs[1,0].plot(xs,series('recovery_triggers'),alpha=.35,color='darkorange');axs[1,0].plot(xs,smooth(series('recovery_triggers')),color='darkorange');axs[1,0].set(title='Recovery triggers',ylabel='Triggers / episode')
    ratio=[100*float(r['recovery_steps'])/max(1,int(r['episode_steps'])) for r in rows]
    axs[1,1].plot(xs,ratio,alpha=.35,color='purple');axs[1,1].plot(xs,smooth(ratio),color='purple');axs[1,1].set(title='Recovery intervention',ylabel='Recovery decisions (%)')
    values=[]
    source=RUN/'recovery_diagnostics.jsonl'
    if source.exists():
        for line in source.open():
            try:ep=json.loads(line)
            except json.JSONDecodeError:continue
            values.extend(float(s['recent_mean_new_free_cell_count']) for s in ep['steps'] if (s['regional_repeat_detected'] or s['point_repeat_detected']) and not s['recovery_mode'])
    if values:
        bins=[0,.125,1,5,10,25,50,100,250,500,1000,5000,max(10000,max(values)+1)]
        axs[1,2].hist(values,bins=bins,color='slategray',edgecolor='white');axs[1,2].set_xscale('symlog',linthresh=1);axs[1,2].axvline(1,color='crimson',ls='--',label='Threshold = 1');axs[1,2].legend()
    axs[1,2].set(title='Repeated visits outside recovery',xlabel='Window mean new free cells / step',ylabel='Observed decisions')
    for ax in axs.flat:ax.grid(alpha=.2)
    for ax in [*axs[0],*axs[1,:2]]:ax.set_xlabel('Completed training episodes')
    tmp=OUT/'training_dashboard.tmp.png';fig.savefig(tmp,dpi=140);plt.close(fig);tmp.replace(OUT/'training_dashboard.png')
    (OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>HERA training</title><body style="font-family:sans-serif;margin:20px"><h2>HERA training: '+str(len(rows))+'/'+str(TOTAL)+' episode</h2><p>每30秒刷新。训练采样曲线，不是正式greedy测试；不据早期随机策略判断阈值优劣。</p><img style="width:100%;max-width:1800px" src="training_dashboard.png?t='+str(time.time_ns())+'"></body></html>')
    print(f'PLOTS_UPDATED {len(rows)} episodes',flush=True)

if __name__=='__main__':
    a=ARGS
    while True:
        draw()
        if not a.watch or (CONTROL/'exit_status.json').exists():break
        time.sleep(30)
