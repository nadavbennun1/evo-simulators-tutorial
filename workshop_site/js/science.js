(function(root,factory){const api=factory();if(typeof module!=="undefined"&&module.exports)module.exports=api;root.WorkshopScience=api})(typeof globalThis!=="undefined"?globalThis:this,function(){"use strict";
  function avecillaDeterministic(theta,generations=120){let p=[1,0,0],out=[p.slice()];const deltaC=10**theta[0],deltaB=10**theta[1],fitness=[1,1+theta[2],1+theta[3]];for(let g=1;g<=generations;g++){const selected=p.map((x,i)=>x*fitness[i]),mutated=[selected[0]*(1-deltaC-deltaB),selected[1]+selected[0]*deltaC,selected[2]+selected[0]*deltaB],total=mutated.reduce((a,b)=>a+b,0);p=mutated.map(x=>x/total);out.push(p.slice())}return out}
  function chuongDeterministic(theta,generations=[8,21,29,37,50,58,66,79,87,95,108,116]){const [logS,logM,logP0]=theta,s=10**logS,m=10**logM,p0=10**logP0,fitness=[1,1+s,1+s,1.001],last=Math.max(...generations);let p=[1-p0,0,p0,0],out=[];for(let g=0;g<=last;g++){if(generations.includes(g))out.push(p[1]);const selected=p.map((x,i)=>x*fitness[i]),mutated=[selected[0]*(1-m-1e-5),selected[1]+selected[0]*m,selected[2],selected[3]+selected[0]*1e-5],total=mutated.reduce((a,b)=>a+b,0);p=mutated.map(x=>x/total)}return out}
  function zhouDeterministic(theta,p0,generations=120){let p=p0.slice(),out=[p.slice()];const muWt=10**theta[0],muLoh=10**theta[3],fitness=[theta[1],1,theta[2]];for(let g=1;g<=generations;g++){const mutated=[p[0]*(1-muWt-muLoh),p[1]+p[0]*muWt,p[2]+p[0]*muLoh],selected=mutated.map((x,i)=>x*fitness[i]),total=selected.reduce((a,b)=>a+b,0);p=selected.map(x=>x/total);out.push(p.slice())}return out}
  function mulberry32(seed){let a=(seed>>>0)||1;return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
  function maskToPassages(mask){const out=[0];for(let p=1;p<=12;p++)if(mask&(1<<(p-1)))out.push(p);return out}
  function collectiveLogPosterior(logPosteriors,logPrior,indices){return logPrior.map((_,j)=>indices.reduce((sum,i)=>sum+logPosteriors[i][j],0)-Math.max(0,indices.length-1)*logPrior[j])}
  function robustCollectiveLogPosterior(logPosteriors,logPrior,indices,logEpsilon){
    const standard=collectiveLogPosterior(logPosteriors,logPrior,indices);
    const robust=logPrior.map((_,j)=>indices.reduce((sum,i)=>sum+Math.max(logEpsilon,logPosteriors[i][j]),0)-Math.max(0,indices.length-1)*logPrior[j]);
    return{standard,robust};
  }
  function erf(value){
    const sign=value<0?-1:1,x=Math.abs(value),t=1/(1+.3275911*x);
    const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-x*x);
    return sign*y;
  }
  function normalCdf(value){return .5*(1+erf(value/Math.sqrt(2)))}
  function effectivePosteriorMean(model,index,strength=1){
    const mean=model.posterior_means[index];
    return index===model.contaminated_index?mean.map((value,k)=>model.truth_theta[k]+strength*(value-model.truth_theta[k])):mean.slice();
  }
  function posteriorLogNormalizer(model,index,mean){
    if(index!==model.contaminated_index)return model.posterior_log_normalizers[index];
    const sd=model.posterior_sds[index],bounds=model.parameter_bounds;
    return bounds.reduce((sum,[lo,hi],k)=>{
      const mass=Math.max(1e-300,normalCdf((hi-mean[k])/sd[k])-normalCdf((lo-mean[k])/sd[k]));
      return sum+Math.log(mass);
    },0);
  }
  function jointPosteriorLog(model,index,theta,strength=1){
    const mean=effectivePosteriorMean(model,index,strength),sd=model.posterior_sds[index];
    let value=-1.5*Math.log(2*Math.PI)-posteriorLogNormalizer(model,index,mean);
    for(let k=0;k<3;k++)value-=Math.log(sd[k])+.5*((theta[k]-mean[k])/sd[k])**2;
    return value;
  }
  function estimateCollectiveLogEpsilon(model,indices,strength=1,q=model.epsilon_calibration.default_quantile){
    const values=[],n=model.epsilon_calibration.prior_draws_per_replicate,bounds=model.parameter_bounds;
    indices.forEach(index=>{
      const random=mulberry32(model.epsilon_calibration.seed+104729*(index+1));
      for(let draw=0;draw<n;draw++){
        const theta=bounds.map(([lo,hi])=>lo+(hi-lo)*random());
        values.push(jointPosteriorLog(model,index,theta,strength));
      }
    });
    values.sort((a,b)=>a-b);
    return values[Math.floor(q*(values.length-1))];
  }
  function collectiveJointSelectionMarginals(model,indices,strength,logEpsilon){
    const [,,nyz]=model.joint_grid_shape,bounds=model.parameter_bounds;
    const ys=Array.from({length:nyz},(_,i)=>bounds[1][0]+i*(bounds[1][1]-bounds[1][0])/(nyz-1));
    const zs=Array.from({length:nyz},(_,i)=>bounds[2][0]+i*(bounds[2][1]-bounds[2][0])/(nyz-1));
    const standard=[],robust=[],correction=Math.max(0,indices.length-1)*model.prior_log_density;
    const accumulate=(state,value)=>value>state.max?{max:value,sum:state.sum*Math.exp(state.max-value)+1}:{max:state.max,sum:state.sum+Math.exp(value-state.max)};
    model.grid.forEach(x=>{
      let standardState={max:-Infinity,sum:0},robustState={max:-Infinity,sum:0};
      ys.forEach(y=>zs.forEach(z=>{
        const theta=[x,y,z],logs=indices.map(index=>jointPosteriorLog(model,index,theta,strength));
        const standardTarget=logs.reduce((sum,value)=>sum+value,0)-correction;
        const robustTarget=logs.reduce((sum,value)=>sum+Math.max(logEpsilon,value),0)-correction;
        standardState=accumulate(standardState,standardTarget);
        robustState=accumulate(robustState,robustTarget);
      }));
      standard.push(standardState.max+Math.log(standardState.sum));
      robust.push(robustState.max+Math.log(robustState.sum));
    });
    return{standard,robust};
  }
  function inverseRmseScore(guess,truth){const rmse=Math.sqrt(guess.reduce((sum,x,i)=>sum+(x-truth[i])**2,0)/guess.length);return{rmse,score:100/(1+rmse)}}
  return{avecillaDeterministic,chuongDeterministic,zhouDeterministic,mulberry32,maskToPassages,collectiveLogPosterior,robustCollectiveLogPosterior,effectivePosteriorMean,jointPosteriorLog,estimateCollectiveLogEpsilon,collectiveJointSelectionMarginals,inverseRmseScore};
});
