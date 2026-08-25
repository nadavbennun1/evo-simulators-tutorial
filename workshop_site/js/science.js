(function(root,factory){const api=factory();if(typeof module!=="undefined"&&module.exports)module.exports=api;root.WorkshopScience=api})(typeof globalThis!=="undefined"?globalThis:this,function(){"use strict";
  function zhouDeterministic(theta,p0,generations=120){let p=p0.slice(),out=[p.slice()];const muWt=10**theta[0],muLoh=10**theta[3],fitness=[theta[1],1,theta[2]];for(let g=1;g<=generations;g++){const mutated=[p[0]*(1-muWt-muLoh),p[1]+p[0]*muWt,p[2]+p[0]*muLoh],selected=mutated.map((x,i)=>x*fitness[i]),total=selected.reduce((a,b)=>a+b,0);p=selected.map(x=>x/total);out.push(p.slice())}return out}
  function mulberry32(seed){let a=(seed>>>0)||1;return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
  function maskToPassages(mask){const out=[0];for(let p=1;p<=12;p++)if(mask&(1<<(p-1)))out.push(p);return out}
  function collectiveLogPosterior(logPosteriors,logPrior,indices){return logPrior.map((_,j)=>indices.reduce((sum,i)=>sum+logPosteriors[i][j],0)-Math.max(0,indices.length-1)*logPrior[j])}
  return{zhouDeterministic,mulberry32,maskToPassages,collectiveLogPosterior};
});
