"use strict";
const fs=require("fs"),path=require("path"),science=require("../js/science.js");
const root=path.resolve(__dirname,"..");
const manifest=JSON.parse(fs.readFileSync(path.join(root,"data/zhou_manifest.json"),"utf8"));
const cases=[
  [[-4,.96,.99,-4.4],[.99,.0075,.0025]],
  [[-3.2,1.01,.94,-5.1],[.8,.15,.05]],
  [[-6,.9,1.04,-3],[1,0,0]],
];
const output=cases.map(([theta,p0])=>science.zhouDeterministic(theta,p0,120));
const avecilla=science.avecillaDeterministic([-4.2,-5,.07,.001],120);
const chuong=science.chuongDeterministic([-.74,-4.84,-4.32]);
const score=science.inverseRmseScore([-.8,-4.7,-4.4],[-.74,-4.84,-4.32]);
const odd=science.maskToPassages(manifest.odd_mask),even=science.maskToPassages(manifest.even_mask);
const lab=JSON.parse(fs.readFileSync(path.join(root,"data/collective_lab.json"),"utf8"));
const collective=science.collectiveLogPosterior(lab.replicate_log_posteriors,lab.prior_log,[0,1,2,3]);
const robustLoose=science.robustCollectiveLogPosterior(lab.replicate_log_posteriors,lab.prior_log,[0,1,2,3],-10);
const robustTight=science.robustCollectiveLogPosterior(lab.replicate_log_posteriors,lab.prior_log,[0,1,2,3],-1000);
const allIndices=lab.labels.map((_,i)=>i),cleanIndices=[0,1,2,3,4];
const estimatedLogEpsilon=science.estimateCollectiveLogEpsilon(lab,allIndices,1,.95);
const epsilonReproducible=estimatedLogEpsilon===science.estimateCollectiveLogEpsilon(lab,allIndices,1,.95);
const neutralContaminantMean=science.effectivePosteriorMean(lab,lab.contaminated_index,0);
const fullContaminantMean=science.effectivePosteriorMean(lab,lab.contaminated_index,1);
const jointEstimated=science.collectiveJointSelectionMarginals(lab,allIndices,1,estimatedLogEpsilon);
const jointFixed=science.collectiveJointSelectionMarginals(lab,allIndices,1,-1000);
const jointClean=science.collectiveJointSelectionMarginals(lab,cleanIndices,1,estimatedLogEpsilon);
const lowEpsilon=science.estimateCollectiveLogEpsilon(lab,allIndices,0,.95);
const highEpsilon=science.estimateCollectiveLogEpsilon(lab,allIndices,1.5,.95);
const jointLowContamination=science.collectiveJointSelectionMarginals(lab,allIndices,0,lowEpsilon);
const jointHighContamination=science.collectiveJointSelectionMarginals(lab,allIndices,1.5,highEpsilon);
const rngA=science.mulberry32(20260825),rngB=science.mulberry32(20260825);
const reproducible=Array.from({length:20},()=>rngA()).every(x=>x===rngB());
process.stdout.write(JSON.stringify({output,avecilla,chuong,score,odd,even,collective,robustLoose,robustTight,estimatedLogEpsilon,epsilonReproducible,neutralContaminantMean,fullContaminantMean,jointEstimated,jointFixed,jointClean,jointLowContamination,jointHighContamination,reproducible}));
