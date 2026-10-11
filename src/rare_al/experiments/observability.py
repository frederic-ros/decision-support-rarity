"""Observability diagnostic (v1 port): can observable region descriptors predict WHERE
the best query lies, and what does acting on them (the regional probe) recover?

``learner`` filters multi-learner oracle outputs (v2); v1 outputs have no learner column.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

ID = {"learner","dataset","domain",
    "regime","ir","d","sep","seed","round","state_id","region_id",
    "contains_point_oracle","is_best_region","region_best_point_gain",
    "region_margin_point_gain","base_auprc",
}

def _features(df):
    return [c for c in df.columns if c not in ID and pd.api.types.is_numeric_dtype(df[c])]

def _matrix(df,cols,med=None):
    X=df[cols].replace([np.inf,-np.inf],np.nan)
    if med is None: med=X.median(numeric_only=True)
    return X.fillna(med).fillna(0.).to_numpy(float),med

def run(cfg,oracle_dir,out_dir,learner=None):
    p=cfg["observability"]
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    states=pd.read_csv(Path(oracle_dir)/"oracle_states.csv")
    regions=pd.read_csv(Path(oracle_dir)/"oracle_regions.csv")
    if learner is not None and "learner" in states:
        states=states[states.learner==learner].copy(); regions=regions[regions.learner==learner].copy()
    threshold=p["oracle_threshold"]
    useful=set(states.loc[states.regional_ceiling_vs_margin>threshold,"state_id"].astype(str))

    fs=_features(regions); pred=[]
    for held in sorted(regions.regime.unique()):
        tr=regions[(regions.regime!=held)&regions.state_id.astype(str).isin(useful)].copy()
        te=regions[regions.regime==held].copy()
        if len(tr)==0 or len(np.unique(tr.contains_point_oracle))<2: continue
        Xtr,med=_matrix(tr,fs);Xte,_=_matrix(te,fs,med)
        clf=ExtraTreesClassifier(
            n_estimators=800,min_samples_leaf=3,max_features=.8,
            class_weight="balanced",random_state=42,n_jobs=-1
        )
        clf.fit(Xtr,tr.contains_point_oracle)
        te["p_region"]=clf.predict_proba(Xte)[:,1];te["heldout_regime"]=held
        pred.append(te)
    pred=pd.concat(pred,ignore_index=True)
    pred.to_csv(out/"region_predictions.csv",index=False)

    smap=states.set_index("state_id"); actions=[]
    for sid,g in pred.groupby("state_id"):
        st=smap.loc[sid]; chosen=g.loc[g.p_region.idxmax()]
        oracle=g.loc[g.region_best_point_gain.idxmax()]
        actions.append({
            "state_id":sid,"regime":st.regime,
            "margin_gain":st.margin_gain,
            "regional_ceiling_vs_margin":st.regional_ceiling_vs_margin,
            "point_ceiling_vs_margin":st.point_ceiling_vs_margin,
            "compass_gain":chosen.region_margin_point_gain,
            "compass_delta_vs_margin":chosen.region_margin_point_gain-st.margin_gain,
            "confidence":chosen.p_region,
            "chosen_is_oracle":int(chosen.region_id==oracle.region_id),
            "useful":int(st.regional_ceiling_vs_margin>threshold),
        })
    actions=pd.DataFrame(actions);actions.to_csv(out/"actions.csv",index=False)

    where_auc=roc_auc_score(pred.contains_point_oracle,pred.p_region)
    when_auc=roc_auc_score(actions.useful,actions.confidence)

    obs=_features(regions)
    agg=regions.groupby("state_id")[obs].agg(["mean","std","min","max"])
    agg.columns=[f"{c}_{s}" for c,s in agg.columns];agg=agg.reset_index()
    sdf=states[["state_id","regime","regional_ceiling_vs_margin"]].merge(agg,on="state_id")
    sdf["target"]=(sdf.regional_ceiling_vs_margin>threshold).astype(int)
    fstate=[c for c in sdf.columns if c not in {"state_id","regime","regional_ceiling_vs_margin","target"}]
    pp=np.full(len(sdf),np.nan)
    for held in sorted(sdf.regime.unique()):
        tr=sdf.regime!=held;te=sdf.regime==held
        Xtr,med=_matrix(sdf.loc[tr],fstate);Xte,_=_matrix(sdf.loc[te],fstate,med)
        clf=ExtraTreesClassifier(
            n_estimators=800,min_samples_leaf=4,class_weight="balanced",
            random_state=43,n_jobs=-1
        )
        clf.fit(Xtr,sdf.loc[tr,"target"]);pp[te]=clf.predict_proba(Xte)[:,1]
    opportunity_auc=roc_auc_score(sdf.target,pp)

    rows=[]
    etas=np.arange(p["eta_start"],p["eta_stop"]+1e-12,p["eta_step"])
    for eta in etas:
        use=actions.confidence>=eta
        delta=np.where(use,actions.compass_delta_vs_margin,0.)
        used=actions[use]
        useful_used=used[used.useful==1]
        non_used=used[used.useful==0]
        rows.append({
            "eta":eta,"coverage":float(use.mean()),"net_mean_delta":float(delta.mean()),
            "rescue":float(useful_used.compass_delta_vs_margin.mean()) if len(useful_used) else np.nan,
            "damage":float(non_used.compass_delta_vs_margin.mean()) if len(non_used) else np.nan,
            "q10":float(used.compass_delta_vs_margin.quantile(.10)) if len(used) else np.nan,
            "q05":float(used.compass_delta_vs_margin.quantile(.05)) if len(used) else np.nan,
            "worst":float(used.compass_delta_vs_margin.min()) if len(used) else np.nan,
        })
    pd.DataFrame(rows).to_csv(out/"compass_sweep.csv",index=False)
    pd.DataFrame([{
        "where_auc":where_auc,
        "region_accuracy":float(actions.chosen_is_oracle.mean()),
        "when_confidence_auc":when_auc,
        "opportunity_gate_auc":opportunity_auc,
        "n_states":len(actions),
    }]).to_csv(out/"observability_summary.csv",index=False)
