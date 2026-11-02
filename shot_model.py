"""Offline development baseline for reviewed shot labels; no pretrained class guesses.

This small NumPy softmax model is a baseline/training interface, not a validated
badminton classifier. Model scores are uncalibrated and can only propose labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from badminton_proto import sha256_file
from contact_proposals import bound_shuttle
from match_analysis import FAMILIES,require,verify,publish
from player_identity import source_snapshot

FEATURE_VERSION='native-observed-five-v1'
FEATURES=('incoming_dx_per_second','incoming_dy_per_second','outgoing_dx_per_second','outgoing_dy_per_second',
          'incoming_speed','outgoing_speed','turn_cosine','contact_x_fraction','contact_y_fraction','window_y_span_fraction')


def features_at(rows,frame,fps,image_size,eligible):
    if type(frame) is not int or frame<2 or frame+2>=len(rows):return None
    indices=list(range(frame-2,frame+3))
    if any(i not in eligible or rows[i].get('proposal') is None or rows[i]['status']!='proposal' for i in indices):return None
    center=rows[frame];segment=center['proposal']['proposal_segment_id'];scene=center['scene_id']
    if any(rows[i]['scene_id']!=scene or rows[i]['proposal']['proposal_segment_id']!=segment for i in indices):return None
    points=np.asarray([rows[i]['proposal']['position_px'] for i in indices],dtype=float)
    width,height=image_size
    require(points.shape==(5,2) and np.isfinite(points).all() and width>0 and height>0 and math.isfinite(fps) and fps>0,'Invalid shot feature geometry')
    require((points>=0).all() and (points[:,0]<width).all() and (points[:,1]<height).all(),'Shot feature observations outside image')
    diagonal=math.hypot(width,height)
    incoming=(points[2]-points[0])*fps/(2*diagonal)
    outgoing=(points[4]-points[2])*fps/(2*diagonal)
    a,b=float(np.linalg.norm(incoming)),float(np.linalg.norm(outgoing))
    cosine=float(np.dot(incoming,outgoing)/(a*b)) if a*b>1e-12 else 0
    return [*incoming.tolist(),*outgoing.tolist(),a,b,max(-1,min(1,cosine)),
            float(points[2,0]/width),float(points[2,1]/height),float(np.ptp(points[:,1])/height)]


def export_examples(job_path,analysis_path,output,split='train',allow_development=False):
    require(not output.exists(),'Choose a new feature output')
    require(split in ('train','validation','test'),'Unknown split')
    packet=verify(job_path,analysis_path)
    require(packet['reviewer_kind']=='human' or (allow_development and packet['reviewer_kind']=='automated'),
            'Reviewed labels are required; automated fixtures need explicit --allow-development-labels')
    context=bound_shuttle(job_path)
    require(context is not None,'No current reviewed-eligible shuttle observations for shot features')
    rows,players,summary,eligible=context
    job,result,_,_,binding=source_snapshot(job_path)
    records=[];skipped=Counter()
    rallies={r['id']:r for r in packet['document']['rallies']}
    for event in packet['document']['events']:
        if event['kind']!='contact' or event['status']!='accepted' or event['shot_status']!='accepted':continue
        if rallies[event['rally_id']]['kind']=='replay' or rallies[event['rally_id']]['status']=='rejected':continue
        if event['start_frame']!=event['end_frame']:
            skipped['uncertain_contact_frame']+=1;continue
        features=features_at(rows,event['start_frame'],packet['source']['fps'],(summary['video']['width_px'],summary['video']['height_px']),eligible)
        if features is None:skipped['missing_observed_context']+=1;continue
        records.append({'example_id':binding['session_id']+':'+event['id'],'session_id':binding['session_id'],
                        'source_video_sha256':binding['source_video_sha256'],'rally_id':event['rally_id'],'event_id':event['id'],
                        'frame_index':event['start_frame'],'reviewer_kind':packet['reviewer_kind'],'label':event['shot_family'],'split':split,'features':features})
    payload={'schema_version':'1.0.0','feature_version':FEATURE_VERSION,'description':'Reviewed ledger labels with unverified model-observation features; development evidence only',
             'analysis_sha256':sha256_file(analysis_path/'analysis.json'),'source_binding':binding,'records':records,'skipped':dict(skipped)}
    require(source_snapshot(job_path)[4]==binding,'Source changed during feature extraction')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return output


def validate_records(dataset,allow_development):
    require(dataset.get('schema_version')=='1.0.0' and dataset.get('feature_version')==FEATURE_VERSION,'Unknown feature dataset schema/version')
    records=dataset.get('records');require(type(records) is list and records,'No training examples')
    seen=set();physical=set();groups={};session_splits={}
    for row in records:
        require(type(row) is dict and set(row)=={'example_id','session_id','source_video_sha256','rally_id','event_id','frame_index','reviewer_kind','label','split','features'},'Invalid training record fields')
        require(type(row['example_id']) is str and row['example_id'] not in seen,'Duplicate example ID');seen.add(row['example_id'])
        require(all(type(row[key]) is str and row[key] for key in ('session_id','rally_id','event_id')),'Missing example identity')
        require(row['label'] in FAMILIES and row['label']!='unknown' and row['split'] in ('train','validation','test'),'Invalid label or split')
        require(row['reviewer_kind']=='human' or (allow_development and row['reviewer_kind']=='automated'),'Automated examples require explicit development mode')
        hash_value=row['source_video_sha256']
        require(type(hash_value) is str and len(hash_value)==64 and all(c in '0123456789abcdef' for c in hash_value),'Invalid example source hash')
        require(type(row['frame_index']) is int and row['frame_index']>=0,'Invalid labeled contact frame')
        key=(hash_value,row['frame_index']);require(key not in physical,'Duplicate physical contact across dataset records');physical.add(key)
        for mapping,key in ((groups,hash_value),(session_splits,row['session_id'])):
            require(key not in mapping or mapping[key]==row['split'],'Source video/session leaks across dataset splits')
            mapping[key]=row['split']
        values=row['features']
        require(type(values) is list and len(values)==len(FEATURES) and all(type(v) in (int,float) and math.isfinite(v) and abs(v)<=1e6 for v in values),'Malformed/nonfinite feature vector')
    return records


def _arrays(model):
    require(model.get('schema_version')=='1.0.0' and model.get('feature_version')==FEATURE_VERSION and model.get('release_status')=='development_only','Unsupported or incorrectly promoted shot model')
    labels=model['classes'];require(type(labels) is list and len(set(labels))==len(labels) and len(labels)>=2 and all(x in FAMILIES and x!='unknown' for x in labels),'Invalid model classes')
    require(tuple(model.get('feature_names',()))==FEATURES,'Model feature order differs')
    mean=np.asarray(model['mean'],dtype=float);scale=np.asarray(model['scale'],dtype=float)
    weights=np.asarray(model['weights'],dtype=float);bias=np.asarray(model['bias'],dtype=float)
    require(mean.shape==scale.shape==(len(FEATURES),) and weights.shape==(len(labels),len(FEATURES)) and bias.shape==(len(labels),),'Invalid model parameter shapes')
    require(all(np.isfinite(v).all() for v in (mean,scale,weights,bias)) and (scale>0).all(),'Nonfinite model parameters')
    return labels,mean,scale,weights,bias


def predict(model,features):
    labels,mean,scale,weights,bias=_arrays(model)
    values=np.asarray(features,dtype=float)
    require(values.shape==(len(FEATURES),) and np.isfinite(values).all() and (np.abs(values)<=1e6).all(),'Invalid prediction features')
    logits=weights@((values-mean)/scale)+bias
    require(np.isfinite(logits).all(),'Model logits are nonfinite')
    scores=np.exp(logits-logits.max());scores/=scores.sum()
    order=np.argsort(-scores);top,second=int(order[0]),int(order[1]);margin=float(scores[top]-scores[second])
    label=labels[top] if scores[top]>=.7 and margin>=.2 else 'unknown'
    return {'shot_family':label,'status':'proposed' if label!='unknown' else 'unknown','score':float(scores[top]),'margin':margin,
            'score_semantics':'uncalibrated softmax output; not a correctness probability','release_status':'development_only'}


def evaluate(model,records):
    counts=Counter();actual=Counter();predicted=Counter();correct=Counter();classified=0
    for row in records:
        result=predict(model,row['features'])['shot_family'];truth=row['label']
        counts[(truth,result)]+=1;actual[truth]+=1;predicted[result]+=1
        correct[truth]+=result==truth;classified+=result!='unknown'
    classes={}
    for label in sorted(actual):
        precision=correct[label]/predicted[label] if predicted[label] else 0
        recall=correct[label]/actual[label]
        classes[label]={'support':actual[label],'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0}
    n=len(records);matched=sum(correct.values())
    return {'examples':n,'classification_coverage':classified/n if n else None,
            'accuracy_all_examples':matched/n if n else None,'selective_accuracy':matched/classified if classified else None,
            'macro_f1':sum(c['f1'] for c in classes.values())/len(classes) if classes else None,
            'per_class':classes,'confusion':[{'actual':a,'predicted':p,'count':v} for (a,p),v in sorted(counts.items())]}


def fit(dataset,minimum=20,allow_development=False):
    records=validate_records(dataset,allow_development)
    require(type(minimum) is int and minimum>=2,'Minimum class support must be at least two')
    train=[r for r in records if r['split']=='train'];test=[r for r in records if r['split']=='test']
    require(len({r['source_video_sha256'] for r in train})>=2 and test,'Need at least two training recordings and a separate test recording')
    labels=sorted({r['label'] for r in train});support=Counter(r['label'] for r in train)
    require(len(labels)>=2 and all(support[label]>=minimum for label in labels),'Insufficient reviewed examples per shot class')
    require(all(r['label'] in labels for r in records),'Evaluation includes a class absent from training')
    x=np.asarray([r['features'] for r in train],dtype=float);y=np.asarray([labels.index(r['label']) for r in train])
    mean=x.mean(axis=0);scale=x.std(axis=0);scale=np.where(scale<1e-8,1,scale);x=(x-mean)/scale
    weights=np.zeros((len(labels),len(FEATURES)));bias=np.zeros(len(labels));truth=np.eye(len(labels))[y]
    for _ in range(400):
        logits=x@weights.T+bias;logits-=logits.max(axis=1,keepdims=True)
        scores=np.exp(logits);scores/=scores.sum(axis=1,keepdims=True);gradient=(scores-truth)/len(x)
        weights-=.05*(gradient.T@x+.001*weights);bias-=.05*gradient.sum(axis=0)
    model={'schema_version':'1.0.0','feature_version':FEATURE_VERSION,'feature_names':FEATURES,'release_status':'development_only',
           'method':'standardized numpy softmax baseline, 400 fixed steps, learning rate 0.05, L2 0.001',
           'classes':labels,'mean':mean.tolist(),'scale':scale.tolist(),'weights':weights.tolist(),'bias':bias.tolist(),
           'review_provenance':dict(Counter(r['reviewer_kind'] for r in records)),
           'training_video_sha256':sorted({r['source_video_sha256'] for r in train}),
           'limitations':['Never auto-accept classes or authorize coaching.','Observed image features are not physical shuttle flight or validated contact geometry.',
                          'Data/model accuracy requires real held-out academy evaluation; synthetic tests are not that evidence.']}
    evaluation={split:evaluate(model,[r for r in records if r['split']==split]) for split in ('train','validation','test')}
    evaluation.update(release_status='development_only',release_gate_passed=False,
                      reason='Training baseline only; confidence calibration, cohort coverage and pilot/coach acceptance are still required')
    return model,evaluation


def train_file(dataset_path,output,minimum=20,allow_development=False):
    require(not output.exists(),'Choose a new model folder')
    raw=dataset_path.read_bytes();dataset=json.loads(raw.decode('utf-8'));model,metrics=fit(dataset,minimum,allow_development)
    model['dataset_sha256']=sha256(raw).hexdigest()
    output.mkdir(parents=True)
    for name,value in (('model.json',model),('evaluation.json',metrics)):
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/'model_manifest.json').write_text(json.dumps({'schema_version':'1.0.0','dataset_sha256':model['dataset_sha256'],
        'artifacts':{name:sha256_file(output/name) for name in ('model.json','evaluation.json')}},indent=2)+'\n',encoding='utf-8')
    return output/'model.json'


def combine_files(inputs,output,allow_development=False):
    require(not output.exists(),'Choose a new combined dataset file')
    records=[];bindings=[]
    for path in inputs:
        data=json.loads(path.read_text(encoding='utf-8'))
        require(data.get('schema_version')=='1.0.0' and data.get('feature_version')==FEATURE_VERSION and type(data.get('records')) is list,
                'Feature input has another schema/version')
        records.extend(data['records']);bindings.append({'filename':path.name,'sha256':sha256_file(path),'examples':len(data['records'])})
    dataset={'schema_version':'1.0.0','feature_version':FEATURE_VERSION,'description':'Combined reviewed feature files; verify cohort/annotation quality independently',
             'source_inputs':bindings,'records':records}
    validate_records(dataset,allow_development)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(dataset,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return output


def load_model(folder):
    manifest=json.loads((folder/'model_manifest.json').read_text(encoding='utf-8'))
    require(manifest['schema_version']=='1.0.0','Unknown model manifest schema')
    for name in ('model.json','evaluation.json'):
        require(sha256_file(folder/name)==manifest['artifacts'][name],'Shot model artifact changed')
    model=json.loads((folder/'model.json').read_text(encoding='utf-8'));_arrays(model)
    require(model.get('dataset_sha256')==manifest.get('dataset_sha256'),'Model dataset binding differs from manifest')
    return model


def propose(job_path,analysis_path,model_folder,output):
    packet=deepcopy(verify(job_path,analysis_path));model=load_model(model_folder)
    base_bytes=(analysis_path/'analysis.json').read_bytes()
    context=bound_shuttle(job_path);predictions={}
    if context is not None:
        rows,players,summary,eligible=context
        for event in packet['document']['events']:
            if event['kind']!='contact' or event['status']=='rejected' or event['shot_status']=='accepted' or event['start_frame']!=event['end_frame']:continue
            features=features_at(rows,event['start_frame'],packet['source']['fps'],(summary['video']['width_px'],summary['video']['height_px']),eligible)
            if features is None:continue
            prediction=predict(model,features);predictions[event['id']]=prediction
            event['shot_family']=prediction['shot_family'];event['shot_status']=prediction['status']
    packet.update(revision=packet['revision']+1,parent_sha256=sha256(base_bytes).hexdigest(),created_utc=datetime.now(timezone.utc).isoformat(),
                  history=packet['history']+[{'revision':packet['revision'],'sha256':sha256(base_bytes).hexdigest(),
                      'reviewer':packet['reviewer'],'reviewer_kind':packet['reviewer_kind'],'operation':'development_shot_proposals'}])
    packet['automatic_shot_model']={'status':'development_proposals' if context is not None else 'not_available',
       'reason':None if context is not None else 'No current reviewed-eligible shuttle context',
       'model_sha256':sha256_file(model_folder/'model.json'),'release_status':'development_only','predictions':predictions,
       'in_training_video':packet['source_binding']['source_video_sha256'] in model['training_video_sha256'],
       'accepted_labels_changed':False}
    require(source_snapshot(job_path)[4]==packet['source_binding'],'Source changed during shot proposals')
    return publish(output,packet,analysis_path/'preview.mp4',parent=base_bytes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('export','combine','train','propose'));parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--job',type=Path);parser.add_argument('--analysis',type=Path);parser.add_argument('--dataset',type=Path)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--input',type=Path,nargs='+')
    parser.add_argument('--split',choices=('train','validation','test'),default='train');parser.add_argument('--minimum-per-class',type=int,default=20)
    parser.add_argument('--allow-development-labels',action='store_true');args=parser.parse_args()
    try:
        if args.command=='export':
            require(args.job is not None and args.analysis is not None,'--job and --analysis are required')
            print(export_examples(args.job,args.analysis,args.out,args.split,args.allow_development_labels))
        elif args.command=='combine':
            require(args.input is not None,'--input feature files are required')
            print(combine_files(args.input,args.out,args.allow_development_labels))
        elif args.command=='train':
            require(args.dataset is not None,'--dataset is required')
            print(train_file(args.dataset,args.out,args.minimum_per_class,args.allow_development_labels))
        else:
            require(args.job is not None and args.analysis is not None and args.model is not None,'--job, --analysis and --model are required')
            print(propose(args.job,args.analysis,args.model,args.out))
    except (ValueError,OSError,KeyError,TypeError) as exc:parser.exit(1,f'Shot model error: {exc}\n')


if __name__=='__main__':main()
