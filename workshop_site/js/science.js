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
  function inverseRmseScore(guess,truth){const rmse=Math.sqrt(guess.reduce((sum,x,i)=>sum+(x-truth[i])**2,0)/guess.length);return{rmse,score:100/(1+rmse)}}
  return{avecillaDeterministic,chuongDeterministic,zhouDeterministic,mulberry32,maskToPassages,collectiveLogPosterior,robustCollectiveLogPosterior,inverseRmseScore};
});
