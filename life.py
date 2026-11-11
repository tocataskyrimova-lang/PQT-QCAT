"""PQT-QCat LIFE-001. Stateful online learner, no offline optimizer/BPTT.

All state mutations have one owner. predict never accepts a target.
Physical W is NOT confused with learned synaptic input/readout matrices.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import argparse,copy,gzip,hashlib,json,math,os,queue,sys,threading,time
import numpy as np

@dataclass
class Config:
    seed:int=11
    core:str='pqt'
    cells:int=6
    alphabet:int=8
    h:float=.01
    integration_substeps:int=1
    event_ticks:int=20
    input_gain:float=2.
    learning_rate:float=.8
    input_learning_rate:float=1e-5
    fast_decay:float=.0005
    consolidation_rate:float=.08
    consolidation_period:int=40
    replay_period:int=20
    replay_capacity:int=64
    learning:bool=True
    replay:bool=True
    consolidation:bool=True
    reset_core:bool=False
    context_gates:bool=True
    internal_drive:bool=True
    guarded_replay:bool=False
    replay_weight_updates:bool=True

class PhysicalCore:
    """Canonical energy, fixed A/Gamma, Heun with explicit subdivision guards."""
    def __init__(self,cfg:Config):
        self.cfg=cfg;self.z=np.zeros((cfg.cells,8),dtype=np.float64)
        self.z[:,6]=.7;self.subdivisions=0
    def grad(self,z):
        x,w,w0,m=z[:,:3],z[:,3:6],z[:,6],z[:,7]
        xn=np.roll(x,-1,axis=-1);xp=np.roll(x,1,axis=-1);th=np.tanh(xn)
        e=x-w*th;r2=np.sum((x-xn)**2,-1)/3;meq=np.exp(-r2/.8)
        gx=x*(x*x-1)+1.5*(e-np.roll(e,1,-1)*np.roll(w,1,-1)*(1-np.tanh(x)**2))
        gx+=2*1.5/(3*.8)*((m-meq)*meq)[:,None]*(2*x-xn-xp)
        gw=1.5*(w-w0[:,None])-1.5*e*th
        g0=1.5*(3*w0-w.sum(-1))+1.2*w0*(w0*w0-1)-.45+1.8*m
        gm=1.5*(m-meq)+1.8*w0
        return np.column_stack((gx,gw,g0,gm))
    def energy(self,z=None):
        z=self.z if z is None else z;x,w,w0,m=z[:,:3],z[:,3:6],z[:,6],z[:,7]
        xn=np.roll(x,-1,-1);e=x-w*np.tanh(xn);meq=np.exp(-np.sum((x-xn)**2,-1)/(3*.8))
        return float(np.sum(.25*np.sum((x*x-1)**2,-1)+.75*np.sum(e*e,-1)+.75*np.sum((w-w0[:,None])**2,-1)+.3*(w0*w0-1)**2-.45*w0+.75*(m-meq)**2+1.8*m*w0))
    def drift(self,z,drive):
        if self.cfg.core=='leaky':
            target=np.zeros_like(z);target[:,:3]=drive
            return -z*np.linspace(.2,2,8)[None,:]+target
        g=self.grad(z);out=-g*np.array([1,1,1,1,1,1,.04,.02])
        if self.cfg.core!='noA':
            J=lambda q:np.roll(q,1,-1)-np.roll(q,-1,-1)
            out[:,:3]+=1.2*J(g[:,:3])+1.5*g[:,3:6]
            out[:,3:6]+=-1.5*g[:,:3]+1.2*J(g[:,3:6])
            out[:,6]-=.35*g[:,7];out[:,7]+=.35*g[:,6]
        out[:,:3]+=drive
        return out
    def _step(self,z,drive,h,depth=0):
        b=self.drift(z,drive);pred=z+h*b
        new=z+.5*h*(b+self.drift(pred,drive))
        err=float(np.max(np.abs(new-pred)))
        if not np.isfinite(new).all() or err>.003:
            if depth>=8:raise FloatingPointError('Integrator could not satisfy local subdivision tolerance')
            self.subdivisions+=1
            return self._step(self._step(z,drive,h/2,depth+1),drive,h/2,depth+1)
        if not np.isfinite(new).all() or np.max(np.abs(new))>100:
            raise FloatingPointError('Invalid physical state; no NaN-to-class conversion')
        return new
    def tick(self,drive):
        for _ in range(self.cfg.integration_substeps):
            self.z=self._step(self.z,drive,self.cfg.h/self.cfg.integration_substeps)

class Life:
    SCHEMA=1
    def __init__(self,cfg:Config|None=None):
        self.cfg=cfg or Config();c=self.cfg
        if c.h<=0 or c.event_ticks<1 or c.integration_substeps<1 or c.core not in ('pqt','noA','leaky'):raise ValueError('Invalid configuration')
        self.rng=np.random.default_rng(c.seed);self.core=PhysicalCore(c)
        self.input_dim=c.alphabet+5
        self.encoder=self.rng.normal(0,.6,(c.cells*3,self.input_dim))
        self.feedback_projection=self.rng.normal(0,1/np.sqrt(c.alphabet),(c.cells*3,c.alphabet))
        self.features=c.cells*16+1
        self.contexts={};self.pending={};self.memory=[];self.event_id=0;self.clock=0
        self.stats={'logical_initializations':1,'predictions':0,'feedback':0,'idle_ticks':0,'ticks':0,
                    'replay_updates':0,'consolidations':0,'new_contexts':0,'expired_eligibility':0,
                    'encoder_update_norm':0.,'readout_update_norm':0.,'core_path_length':0.,'external_events':0,'core_resets':0,
                    'replay_proposals':0,'replay_rejections':0,'guard_anchor_evaluations':0,'internal_replay_events':0}
        self.last_internal_drive=np.zeros((c.cells,3));self.last_mode='initialized'
        self.guard_cache={};self.guard_cache_clock=-1000000
    def _key(self,context):return str(context) if self.cfg.context_gates else '__shared__'
    def _context(self,context):
        key=self._key(context)
        if key not in self.contexts:
            if len(self.contexts)>=32:raise ValueError('Context capacity exhausted')
            self.contexts[key]={'fast':np.zeros((self.cfg.alphabet,self.features)),
                'slow':np.zeros((self.cfg.alphabet,self.features)),'observations':0}
            self.stats['new_contexts']+=1
        return self.contexts[key]
    def _encode(self,token,context):
        if not isinstance(token,(int,np.integer)) or not 0<=token<self.cfg.alphabet:raise ValueError('Token outside alphabet')
        # Stable, target-independent context embedding. Contexts are explicitly observed.
        digest=hashlib.sha256(str(context).encode()).digest()
        emb=np.frombuffer(digest[:4],np.uint8).astype(float)/127.5-1
        u=np.zeros(self.input_dim);u[token]=1;u[self.cfg.alphabet:self.cfg.alphabet+4]=.3*emb;u[-1]=1
        return u
    def _drive(self,u):return self.cfg.input_gain*np.tanh(self.encoder@u).reshape(self.cfg.cells,3)
    @staticmethod
    def _softmax(v):
        if not np.isfinite(v).all():raise FloatingPointError('Nonfinite logits; prediction rejected')
        v=v-v.max();p=np.exp(v);return p/p.sum()
    def _prob(self,context,e):
        block=self._context(context);return self._softmax((block['fast']+block['slow'])@e)
    def consolidate(self):
        if not self.cfg.learning or not self.cfg.consolidation:return
        for block in self.contexts.values():
            transfer=self.cfg.consolidation_rate*block['fast']
            block['slow']+=transfer;block['fast']-=transfer
        self.stats['consolidations']+=1
    def _readout_update(self,context,e,target,lr):
        p=self._prob(context,e);err=-p;err[target]+=1
        delta=lr*np.outer(err,e)/(np.dot(e,e)+1e-8)
        self._context(context)['fast']+=delta
        self.stats['readout_update_norm']+=float(np.linalg.norm(delta))
        return err
    @staticmethod
    def _features_from_states(before,after,duration):
        delta=(after-before)/duration
        e=np.r_[.15*np.tanh(after).ravel(),2*np.tanh(delta).ravel(),1.]
        return e/max(np.linalg.norm(e),1e-8)
    def _refresh_guard(self):
        """Past-labelled anchors, re-encoded by a shadow core; never a new teacher.
        Covers at most one recent record per observed context/token. This is a
        proxy guard, not a guarantee for unseen states or future rules.
        """
        chosen={}
        for item in sorted(self.memory,key=lambda r:r.get('received_at',0)):
            chosen[(item['context'],item['token'])]=item
        cache={};shadow=PhysicalCore(self.cfg);shadow.z=self.core.z.copy()
        for (context,token),item in sorted(chosen.items()):
            before=shadow.z.copy();force=self._drive(self._encode(token,context))
            for _ in range(self.cfg.event_ticks):shadow.tick(force)
            e=self._features_from_states(before,shadow.z,self.cfg.h*self.cfg.event_ticks)
            cache.setdefault(self._key(context),[]).append((e,item['target']))
            self.stats['guard_anchor_evaluations']+=1
        self.guard_cache={k:{'x':np.stack([q[0] for q in v]),'y':np.array([q[1] for q in v],dtype=np.int64)} for k,v in cache.items()}
        self.guard_cache_clock=self.clock
    def _guard_score(self,key):
        a=self.guard_cache[key];block=self.contexts[key];logits=a['x']@(block['fast']+block['slow']).T
        shifted=logits-logits.max(1,keepdims=True);lognorm=np.log(np.exp(shifted).sum(1))
        loss=np.mean(lognorm-shifted[np.arange(len(shifted)),a['y'].astype(int)])
        correct=int((logits.argmax(1)==a['y']).sum())
        return float(loss),correct
    def _replay_update(self,item):
        self.stats['replay_proposals']+=1
        key=self._key(item['context']);old=None;before=None
        if self.cfg.guarded_replay:
            if self.clock-self.guard_cache_clock>=200 or key not in self.guard_cache:self._refresh_guard()
            if key in self.guard_cache:
                before=self._guard_score(key);old=self.contexts[key]['fast'].copy()
        norm_before=self.stats['readout_update_norm']
        self._readout_update(item['context'],item['eligibility'],item['target'],self.cfg.learning_rate*.25)
        if old is not None:
            after=self._guard_score(key)
            if after[0]>before[0]+1e-10 or after[1]<before[1]:
                self.contexts[key]['fast']=old;self.stats['replay_rejections']+=1
                self.stats['readout_update_norm']=norm_before
                return False
        self.stats['replay_updates']+=1
        return True
    def tick(self,drive=None,idle=False):
        self.clock+=1;self.stats['ticks']+=1
        if idle:self.stats['idle_ticks']+=1
        if self.cfg.learning:
            for b in self.contexts.values():b['fast']*=1-self.cfg.fast_decay
        if self.clock%self.cfg.consolidation_period==0:self.consolidate()
        if idle and self.cfg.learning and self.cfg.replay and self.memory and self.clock%self.cfg.replay_period==0:
            item=self.memory[int(self.rng.integers(len(self.memory)))]
            self.stats['internal_replay_events']+=1
            if self.cfg.replay_weight_updates:self._replay_update(item)
            self.last_internal_drive=self._drive(self._encode(item['token'],item['context']))*.3
        force=np.zeros((self.cfg.cells,3)) if drive is None else drive
        if idle and self.cfg.internal_drive:force=self.last_internal_drive
        previous=self.core.z.copy();self.core.tick(force)
        self.stats['core_path_length']+=float(np.linalg.norm(self.core.z-previous))
        self.last_mode='internal' if idle else 'interaction'
    def idle(self,ticks):
        if not isinstance(ticks,int) or ticks<0:raise ValueError('Invalid ticks')
        for _ in range(ticks):self.tick(idle=True)
    def predict(self,token,context='default'):
        u=self._encode(token,context);self._context(context)
        if self.cfg.reset_core:
            self.core=PhysicalCore(self.cfg) # explicit negative-control only
            self.stats['core_resets']+=1
        before=self.core.z.copy();drive=self._drive(u)
        for _ in range(self.cfg.event_ticks):self.tick(drive)
        e=self._features_from_states(before,self.core.z,self.cfg.h*self.cfg.event_ticks)
        probs=self._prob(context,e);pred=int(np.argmax(probs))
        self.event_id+=1;eid=self.event_id
        inp_trace=(1-np.tanh(self.encoder@u)**2)[:,None]*u[None,:]
        self.pending[eid]={'context':str(context),'token':int(token),'eligibility':e.copy(),'input_trace':inp_trace,'clock':self.clock}
        if len(self.pending)>256:
            del self.pending[min(self.pending)];self.stats['expired_eligibility']+=1
        self._context(context)['observations']+=1;self.stats['predictions']+=1;self.stats['external_events']+=1
        return {'event_id':eid,'token':pred,'probabilities':probs.tolist(),'tick':self.clock}
    def feedback(self,event_id,target):
        if not isinstance(target,(int,np.integer)) or not 0<=target<self.cfg.alphabet:raise ValueError('Invalid feedback target')
        if event_id not in self.pending:raise KeyError('Unknown, expired, or already consumed prediction')
        item=self.pending.pop(event_id);self.stats['feedback']+=1
        if not self.cfg.learning:return
        err=self._readout_update(item['context'],item['eligibility'],target,self.cfg.learning_rate)
        # Approximate local eligibility + fixed feedback alignment, NOT exact e-prop/RTRL.
        decay=math.exp(-.001*(self.clock-item['clock']))
        update=self.cfg.input_learning_rate*decay*(self.feedback_projection@err)[:,None]*item['input_trace']
        self.encoder+=update
        rownorm=np.linalg.norm(self.encoder,axis=1,keepdims=True);self.encoder/=np.maximum(1,rownorm/4)
        self.stats['encoder_update_norm']+=float(np.linalg.norm(update))
        record={'context':item['context'],'token':item['token'],'eligibility':item['eligibility'],'target':int(target),'received_at':self.clock}
        # Bounded replay reservoir per explicit context; no table lookup during prediction.
        same=[i for i,v in enumerate(self.memory) if v['context']==item['context']]
        if len(same)<self.cfg.replay_capacity:self.memory.append(record)
        else:self.memory[int(self.rng.choice(same))]=record
    def generate(self,start,context='default',length=8):
        if length<0:raise ValueError('Invalid length')
        token=start;result=[]
        for _ in range(length):
            answer=self.predict(token,context);result.append(answer);token=answer['token']
        return result
    def snapshot(self):
        return {'schema':self.SCHEMA,'config':asdict(self.cfg),'z':self.core.z.copy(),'subdivisions':self.core.subdivisions,
                'rng':copy.deepcopy(self.rng.bit_generator.state),'encoder':self.encoder.copy(),'feedback_projection':self.feedback_projection.copy(),
                'contexts':copy.deepcopy(self.contexts),'pending':copy.deepcopy(self.pending),'memory':copy.deepcopy(self.memory),
                'event_id':self.event_id,'clock':self.clock,'stats':copy.deepcopy(self.stats),
                'last_internal_drive':self.last_internal_drive.copy(),'last_mode':self.last_mode,
                'guard_cache':copy.deepcopy(self.guard_cache),'guard_cache_clock':self.guard_cache_clock}
    @classmethod
    def restore(cls,state):
        if state['schema']!=cls.SCHEMA:raise ValueError('Unsupported snapshot schema')
        life=cls(Config(**state['config']));life.core.z=state['z'].copy();life.core.subdivisions=state['subdivisions']
        life.rng.bit_generator.state=copy.deepcopy(state['rng'])
        for key in ['encoder','feedback_projection','contexts','memory','event_id','clock','last_internal_drive','last_mode']:
            setattr(life,key,copy.deepcopy(state[key]))
        life.stats.update(copy.deepcopy(state['stats']))
        life.guard_cache=copy.deepcopy(state.get('guard_cache',{}));life.guard_cache_clock=state.get('guard_cache_clock',-1000000)
        life.pending={int(k):copy.deepcopy(v) for k,v in state['pending'].items()}
        return life
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
        def encode(x):
            if isinstance(x,np.ndarray):return {'__array__':x.tolist()}
            if isinstance(x,np.generic):return x.item()
            raise TypeError(type(x))
        with gzip.open(tmp,'wt',encoding='utf8') as f:json.dump(self.snapshot(),f,default=encode,allow_nan=False)
        os.replace(tmp,path)
    @classmethod
    def load(cls,path):
        def decode(x):return np.array(x['__array__'],dtype=np.float64) if set(x)=={'__array__'} else x
        with gzip.open(path,'rt',encoding='utf8') as f:state=json.load(f,object_hook=decode)
        return cls.restore(state)
    def status(self):
        return {**self.stats,'clock':self.clock,'event_id':self.event_id,'contexts':list(self.contexts),
                'memory_items':len(self.memory),'pending_feedback':len(self.pending),
                'finite':bool(np.isfinite(self.core.z).all()),'max_abs_state':float(np.max(np.abs(self.core.z))),
                'physical_energy':self.core.energy(),'subdivisions':self.core.subdivisions,
                'encoder_norm':float(np.linalg.norm(self.encoder)),
                'fast_norm':float(sum(np.linalg.norm(v['fast']) for v in self.contexts.values())),
                'slow_norm':float(sum(np.linalg.norm(v['slow']) for v in self.contexts.values()))}

def serve(args):
    life=Life.load(args.checkpoint) if args.checkpoint and Path(args.checkpoint).exists() else Life(Config(seed=args.seed))
    if args.guarded_replay:life.cfg.guarded_replay=True
    if args.state_only_replay:life.cfg.replay_weight_updates=False
    inbox=queue.Queue()
    def reader():
        for line in sys.stdin:inbox.put(line)
        inbox.put(None)
    threading.Thread(target=reader,daemon=True).start()
    print(json.dumps({'ready':True,'status':life.status()}),flush=True)
    while True:
        try:line=inbox.get(timeout=1/args.idle_hz)
        except queue.Empty:life.idle(1);continue
        if line is None:break
        try:
            request=json.loads(line);op=request.get('op','predict')
            if op=='predict':reply=life.predict(request['token'],request.get('context','default'))
            elif op=='feedback':life.feedback(request['event_id'],request['target']);reply={'accepted':True}
            elif op=='generate':reply=life.generate(request['token'],request.get('context','default'),request.get('length',8))
            elif op=='idle':life.idle(request['ticks']);reply=life.status()
            elif op=='status':reply=life.status()
            elif op=='save':life.save(request.get('path',args.checkpoint or 'life.json.gz'));reply={'saved':True}
            elif op=='quit':break
            else:raise ValueError('Unknown operation')
            print(json.dumps(reply),flush=True)
        except (ValueError,KeyError,TypeError) as e:print(json.dumps({'error':str(e)}),flush=True)
    if args.checkpoint:life.save(args.checkpoint)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--serve',action='store_true');parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--checkpoint');parser.add_argument('--idle-hz',type=float,default=20)
    parser.add_argument('--guarded-replay',action='store_true')
    parser.add_argument('--state-only-replay',action='store_true')
    args=parser.parse_args()
    if args.idle_hz<=0:parser.error('--idle-hz must be positive')
    if args.serve:serve(args)
    else:print('Use --serve for a persistent JSON-lines process. See README.md.')
