(function(root){
  "use strict";

  const Science=root.WorkshopScience;
  const COLORS={blue:"#829ead",blueDark:"#38596b",blueDeep:"#183743",terra:"#c96b4b",terraDark:"#8f412e",grid:"#d8dfdf",muted:"#60727a",paper:"#fffefa"};
  const ABC_BOUNDS={s:[0,.14],logMu:[-6,-2.8]};
  const ABC_TRUTH={s:.055,logMu:-4.15};
  const ABC_TIMES=Array.from({length:11},(_,i)=>i*10);

  function rng(seed){
    if(Science&&Science.mulberry32)return Science.mulberry32(seed);
    let a=(seed>>>0)||1;
    return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296};
  }

  function evolveAllele(s,logMu,times=ABC_TIMES){
    const mu=10**logMu,last=times[times.length-1],wanted=new Set(times);
    let p=.02;
    const out=[];
    for(let t=0;t<=last;t++){
      if(wanted.has(t))out.push(p);
      const mutated=p+mu*(1-p);
      p=(mutated*(1+s))/(mutated*(1+s)+(1-mutated));
    }
    return out;
  }

  function observedTrajectory(){
    const random=rng(4729),clean=evolveAllele(ABC_TRUTH.s,ABC_TRUTH.logMu);
    return clean.map((value,i)=>Math.max(0,Math.min(1,value+(random()-.5)*(.016+i*.0012))));
  }

  const OBSERVED=observedTrajectory();
  function distance(a,b){return Math.sqrt(a.reduce((sum,value,i)=>sum+(value-b[i])**2,0)/a.length)}
  function proposal(random){
    const s=ABC_BOUNDS.s[0]+random()*(ABC_BOUNDS.s[1]-ABC_BOUNDS.s[0]);
    const logMu=ABC_BOUNDS.logMu[0]+random()*(ABC_BOUNDS.logMu[1]-ABC_BOUNDS.logMu[0]);
    const trajectory=evolveAllele(s,logMu);
    return{s,logMu,trajectory,distance:distance(trajectory,OBSERVED)};
  }
  function quantile(values,q){
    const sorted=values.slice().sort((a,b)=>a-b);
    if(!sorted.length)return NaN;
    const at=(sorted.length-1)*q,lo=Math.floor(at),hi=Math.ceil(at);
    return sorted[lo]+(sorted[hi]-sorted[lo])*(at-lo);
  }
  function abcBatch(seed,budget,acceptPct){
    const random=rng(seed),samples=[];
    for(let i=0;i<budget;i++)samples.push(proposal(random));
    samples.sort((a,b)=>a.distance-b.distance);
    return{samples,accepted:samples.slice(0,Math.max(1,Math.round(budget*acceptPct/100)))};
  }

  function context2d(canvas){
    const ratio=Math.min(2,root.devicePixelRatio||1);
    const rect=canvas.getBoundingClientRect();
    const width=Math.max(320,Math.round(rect.width||canvas.width));
    const height=Math.max(220,Math.round(width*(canvas.height/canvas.width)));
    if(canvas.width!==Math.round(width*ratio)||canvas.height!==Math.round(height*ratio)){
      canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);
    }
    const ctx=canvas.getContext("2d");
    ctx.setTransform(ratio,0,0,ratio,0,0);
    return{ctx,width,height};
  }
  function clearCanvas(canvas){const {ctx,width,height}=context2d(canvas);ctx.clearRect(0,0,width,height);return{ctx,width,height}}
  function plotFrame(ctx,width,height,xLabel,yLabel,padding={left:58,right:20,top:18,bottom:48}){
    const x0=padding.left,x1=width-padding.right,y0=height-padding.bottom,y1=padding.top;
    ctx.strokeStyle=COLORS.grid;ctx.lineWidth=1;
    for(let i=0;i<=4;i++){const y=y0-(y0-y1)*i/4;ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(x1,y);ctx.stroke()}
    ctx.strokeStyle=COLORS.blueDeep;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x0,y1);ctx.lineTo(x0,y0);ctx.lineTo(x1,y0);ctx.stroke();
    ctx.fillStyle=COLORS.muted;ctx.font="600 12px Inter, sans-serif";ctx.textAlign="center";ctx.fillText(xLabel,(x0+x1)/2,height-9);
    ctx.save();ctx.translate(14,(y0+y1)/2);ctx.rotate(-Math.PI/2);ctx.fillText(yLabel,0,0);ctx.restore();
    return{x0,x1,y0,y1};
  }
  function line(ctx,points,mapX,mapY,color,width=2,alpha=1,dash=[]){
    if(!points.length)return;
    ctx.save();ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.lineJoin="round";ctx.lineCap="round";ctx.beginPath();
    points.forEach((point,i)=>{const x=mapX(point[0]),y=mapY(point[1]);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke();ctx.restore();
  }
  function dot(ctx,x,y,r,color,alpha=1){ctx.save();ctx.globalAlpha=alpha;ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();ctx.restore()}

  const story={
    titles:["Generate possible worlds","Learn the inverse map","Read uncertainty, not only a point"],
    texts:[
      "Draw θ from the prior, run the mechanistic model, and repeat. Each parameter setting produces a possible evolutionary trajectory.",
      "Train a conditional density estimator on paired parameters and simulations. It learns which θ are compatible with a trajectory.",
      "Give the observed data to the trained model. The posterior shows the parameter values that remain plausible and how uncertain they are."
    ]
  };
  let storyStep=0,storyTimer=null;

  function buildNetwork(){
    const svg=document.querySelector(".neural-panel svg"),lineGroup=svg&&svg.querySelector(".net-lines"),nodeGroup=svg&&svg.querySelector(".net-nodes");
    if(!svg||lineGroup.children.length)return;
    const layers=[[35,[32,68,104,140]],[112,[22,55,88,121,154]],[190,[40,77,114,151]],[242,[61,91,121]]];
    const ns="http://www.w3.org/2000/svg";
    layers.slice(0,-1).forEach((layer,i)=>layer[1].forEach(y=>layers[i+1][1].forEach(y2=>{const path=document.createElementNS(ns,"line");path.setAttribute("x1",layer[0]);path.setAttribute("y1",y);path.setAttribute("x2",layers[i+1][0]);path.setAttribute("y2",y2);path.setAttribute("class","net-link");lineGroup.append(path)})));
    layers.forEach((layer,i)=>layer[1].forEach((y,j)=>{const circle=document.createElementNS(ns,"circle");circle.setAttribute("cx",layer[0]);circle.setAttribute("cy",y);circle.setAttribute("r",i===1?8:9);circle.setAttribute("class",`net-node${(i+j)%4===0?" hot":""}`);nodeGroup.append(circle)}));
  }

  function drawStoryTrajectories(){
    const canvas=document.getElementById("story-trajectories");if(!canvas)return;
    const {ctx,width,height}=clearCanvas(canvas),f=plotFrame(ctx,width,height,"Time","Allele frequency",{left:64,right:22,top:20,bottom:52});
    ctx.fillStyle=COLORS.muted;ctx.font="600 11px Inter, sans-serif";ctx.textAlign="right";
    [0,.25,.5,.75,1].forEach(v=>ctx.fillText(v.toFixed(v===0||v===1?0:2),f.x0-9,f.y0-(f.y0-f.y1)*v+4));
    ctx.textAlign="center";[0,25,50,75,100].forEach(v=>ctx.fillText(String(v),f.x0+(f.x1-f.x0)*v/100,f.y0+18));
    const random=rng(981);
    for(let i=0;i<13;i++){
      const s=.012+random()*.085,logMu=-5.7+random()*2.4,values=evolveAllele(s,logMu);
      line(ctx,values.map((v,j)=>[ABC_TIMES[j],v]),x=>f.x0+(f.x1-f.x0)*x/100,y=>f.y0-(f.y0-f.y1)*y,i%4===0?COLORS.terra:COLORS.blue,i%4===0?2.3:1.5,.28+i*.03);
    }
  }

  function setStoryStep(next){
    storyStep=(next+3)%3;
    document.querySelectorAll("[data-story-step]").forEach((button,i)=>{button.setAttribute("aria-selected",String(i===storyStep));button.tabIndex=i===storyStep?0:-1});
    const visual=document.querySelector(".story-visual");if(visual)visual.dataset.activeStep=String(storyStep);
    const number=document.getElementById("story-number"),title=document.getElementById("story-title"),text=document.getElementById("story-text");
    if(number)number.textContent=`STEP 0${storyStep+1}`;if(title)title.textContent=story.titles[storyStep];if(text)text.textContent=story.texts[storyStep];
    if(storyStep===0)drawStoryTrajectories();
  }
  function stopStory(){
    if(storyTimer)root.clearInterval(storyTimer);storyTimer=null;
    const play=document.getElementById("story-play");if(play){play.setAttribute("aria-pressed","false");play.innerHTML='<span aria-hidden="true">▶</span> Play the three steps'}
  }
  function toggleStory(){
    const play=document.getElementById("story-play");
    if(storyTimer){stopStory();return}
    setStoryStep(storyStep+1);storyTimer=root.setInterval(()=>setStoryStep(storyStep+1),2200);
    play.setAttribute("aria-pressed","true");play.innerHTML='<span aria-hidden="true">Ⅱ</span> Pause animation';
  }

  let abcRunId=0,lastAbc=null;
  function drawAbcTrajectories(proposals=[],accepted=[]){
    const canvas=document.getElementById("abc-trajectories");if(!canvas)return;
    const {ctx,width,height}=clearCanvas(canvas),f=plotFrame(ctx,width,height,"Time","Allele frequency",{left:58,right:18,top:18,bottom:48});
    const mapX=x=>f.x0+(f.x1-f.x0)*x/100,mapY=y=>f.y0-(f.y0-f.y1)*y;
    ctx.fillStyle=COLORS.muted;ctx.font="600 10px Inter, sans-serif";ctx.textAlign="right";[0,.5,1].forEach(v=>ctx.fillText(v.toFixed(v===.5?1:0),f.x0-8,mapY(v)+4));
    ctx.textAlign="center";[0,25,50,75,100].forEach(v=>ctx.fillText(String(v),mapX(v),f.y0+17));
    proposals.slice(-55).forEach(sample=>line(ctx,sample.trajectory.map((v,i)=>[ABC_TIMES[i],v]),mapX,mapY,"#bfc8c9",1,.2));
    accepted.slice(0,80).reverse().forEach(sample=>line(ctx,sample.trajectory.map((v,i)=>[ABC_TIMES[i],v]),mapX,mapY,COLORS.blueDark,1.2,.16));
    line(ctx,OBSERVED.map((v,i)=>[ABC_TIMES[i],v]),mapX,mapY,COLORS.terra,3,1);
    OBSERVED.forEach((v,i)=>dot(ctx,mapX(ABC_TIMES[i]),mapY(v),3.8,COLORS.terra));
  }
  function drawAbcPosterior(samples=[],accepted=[]){
    const canvas=document.getElementById("abc-posterior");if(!canvas)return;
    const {ctx,width,height}=clearCanvas(canvas),f=plotFrame(ctx,width,height,"Selection coefficient, s","log₁₀ mutation rate, μ",{left:67,right:18,top:18,bottom:48});
    const mapX=x=>f.x0+(f.x1-f.x0)*(x-ABC_BOUNDS.s[0])/(ABC_BOUNDS.s[1]-ABC_BOUNDS.s[0]);
    const mapY=y=>f.y0-(f.y0-f.y1)*(y-ABC_BOUNDS.logMu[0])/(ABC_BOUNDS.logMu[1]-ABC_BOUNDS.logMu[0]);
    ctx.fillStyle=COLORS.muted;ctx.font="600 10px Inter, sans-serif";ctx.textAlign="center";[0,.035,.07,.105,.14].forEach(v=>ctx.fillText(v.toFixed(3).replace(/0+$/,""),mapX(v),f.y0+17));
    ctx.textAlign="right";[-6,-5.2,-4.4,-3.6,-2.8].forEach(v=>ctx.fillText(v.toFixed(1),f.x0-9,mapY(v)+4));
    samples.filter((_,i)=>i%Math.max(1,Math.floor(samples.length/550))===0).forEach(sample=>dot(ctx,mapX(sample.s),mapY(sample.logMu),1.8,"#c5cdce",.35));
    accepted.forEach(sample=>dot(ctx,mapX(sample.s),mapY(sample.logMu),3,COLORS.blueDark,.48));
    if(accepted.length){
      const sx=quantile(accepted.map(d=>d.s),.5),my=quantile(accepted.map(d=>d.logMu),.5);
      ctx.strokeStyle=COLORS.terra;ctx.lineWidth=2.5;ctx.beginPath();ctx.arc(mapX(sx),mapY(my),8,0,Math.PI*2);ctx.stroke();
    }
  }
  function setAbcSummary(accepted){
    const box=document.getElementById("abc-summary");if(!box)return;
    if(!accepted.length){box.innerHTML='<span><small>Accepted</small><b>—</b></span><span><small>Posterior median s</small><b>—</b></span><span><small>Posterior median log₁₀ μ</small><b>—</b></span><p>Run the simulator to turn prior guesses into a posterior approximation.</p>';return}
    const s=accepted.map(d=>d.s),m=accepted.map(d=>d.logMu),s50=quantile(s,.5),m50=quantile(m,.5),s10=quantile(s,.1),s90=quantile(s,.9),m10=quantile(m,.1),m90=quantile(m,.9);
    box.innerHTML=`<span><small>Accepted</small><b>${accepted.length}</b></span><span><small>Posterior median s</small><b>${s50.toFixed(3)}</b></span><span><small>Posterior median log₁₀ μ</small><b>${m50.toFixed(2)}</b></span><p>Middle 80%: s ${s10.toFixed(3)}–${s90.toFixed(3)}; log₁₀ μ ${m10.toFixed(2)}–${m90.toFixed(2)}. Fewer accepted simulations sharpen the cloud but increase Monte Carlo noise.</p>`;
  }
  function runAbc(){
    const budget=Number(document.getElementById("abc-sims").value),acceptPct=Number(document.getElementById("abc-accept").value),runId=++abcRunId,random=rng(12031+budget*17+acceptPct),samples=[];
    const progress=document.getElementById("abc-progress"),label=document.getElementById("abc-progress-label"),count=document.getElementById("abc-progress-count"),button=document.getElementById("abc-run");
    progress.max=budget;progress.value=0;label.textContent="Simulating…";button.disabled=true;button.textContent="Building possible worlds…";setAbcSummary([]);drawAbcPosterior([],[]);
    const chunk=Math.max(50,Math.ceil(budget/36));
    function frame(){
      if(runId!==abcRunId)return;
      const end=Math.min(budget,samples.length+chunk);while(samples.length<end)samples.push(proposal(random));
      progress.value=samples.length;count.textContent=`${samples.length.toLocaleString()} / ${budget.toLocaleString()}`;drawAbcTrajectories(samples,[]);
      if(samples.length<budget){root.requestAnimationFrame(frame);return}
      const ranked=samples.slice().sort((a,b)=>a.distance-b.distance),accepted=ranked.slice(0,Math.max(1,Math.round(budget*acceptPct/100)));
      lastAbc={samples,accepted};drawAbcTrajectories(samples,accepted);drawAbcPosterior(samples,accepted);setAbcSummary(accepted);
      label.textContent="Posterior ready";button.disabled=false;button.textContent="Run again";
    }
    root.requestAnimationFrame(frame);
  }
  function resetAbc(){
    abcRunId++;lastAbc=null;const progress=document.getElementById("abc-progress"),label=document.getElementById("abc-progress-label"),button=document.getElementById("abc-run");
    progress.value=0;label.textContent="Ready";document.getElementById("abc-progress-count").textContent=`0 / ${Number(document.getElementById("abc-sims").value).toLocaleString()}`;button.disabled=false;button.textContent="Run rejection ABC";setAbcSummary([]);drawAbcTrajectories();drawAbcPosterior();
  }

  const caseStudies={
    candida:{
      kicker:"CANDIDA · ANEUPLOIDY REVERSION",
      title:"Which routes return an aneuploid population to euploidy?",
      context:"Replicate populations begin with an extra chromosome copy. The model separates direct chromosome loss from loss of heterozygosity, while allowing each state to grow at a different rate.",
      question:"Which chromosome-specific reversion rates and relative fitnesses can jointly explain the observed frequency trajectories?",
      tag:"MAP estimates",
      steps:["Begin with a mostly aneuploid population.","Direct chromosome loss produces the euploid wild-type state.","A second route creates loss of heterozygosity; selection then reshapes all three frequencies."],
      labels:["Chr4","Chr5","Chr6","Chr7"],
      valuesA:[.00009824,.00039690,.00139748,.00179746],
      valuesB:[.00015012,.00033596,.00097374,.00211214],
      max:.00225,
      series:["BFP","GFP"],
      findings:[
        ["~16× range","The inferred HET→WT rate spans roughly sixteen-fold from chromosome 4 to chromosome 7."],
        ["Replicates agree","BFP and GFP estimates preserve the same broad chromosome ordering despite experimental variation."]
      ]
    },
    ms2:{
      kicker:"MS2 · PUBLIC-GOODS DYNAMICS",
      title:"When can one phage genome rescue another?",
      context:"At low multiplicity of infection (MOI), a mutation largely experiences its own fitness effect. At high MOI, genomes share a host cell, so gene products can complement recessive defects.",
      question:"Can one mechanistic model explain how mutation effects change between mostly single infection and frequent coinfection?",
      tag:"Collective posterior MAP",
      steps:["At low MOI, most cells receive one genome and mutation effects are directly exposed.","At high MOI, multiple genomes often share the same host cell.","Shared gene products can complement recessive defects, changing which mutants persist."],
      labels:["mat","cp","lys","rep"],
      valuesA:[.80988,.69413,.71437,.77031],
      valuesB:[.24056,.34552,.91594,.31331],
      max:1,
      series:["Low-MOI nonsynonymous fitness","High-MOI recessive probability"],
      findings:[
        ["Gene-specific costs","At low MOI, inferred nonsynonymous fitness differs across the four MS2 genes."],
        ["Lysis stands out","At high MOI, lysis mutations have the strongest inferred recessive/complementable signal (≈0.92)."]
      ]
    }
  };
  let currentCase="candida",caseStep=0,caseTimer=null;
  function caseModelMarkup(name){
    if(name==="candida")return '<div class="candida-diagram"><span class="candida-state aneu">Aneu</span><i class="candida-edge one"></i><span class="candida-state">WT</span><i class="candida-edge two"></i><span class="candida-state loh">LOH</span><b class="traveller"></b></div><p class="model-caption"></p>';
    return '<div class="ms2-diagram"><div class="moi-world low"><strong>Low MOI</strong><span class="host-cell"></span><i class="phage-dot"></i></div><div class="moi-world high"><strong>High MOI</strong><span class="host-cell"></span><i class="phage-dot"></i><i class="phage-dot"></i><i class="phage-dot"></i><span class="public-good"></span></div></div><p class="model-caption"></p>';
  }
  function resultMarkup(study){
    const bars=study.labels.map((label,i)=>`<div class="bar-group"><i class="bar" style="--value:${study.valuesA[i]/study.max}"><em>${study.valuesA[i]<.01?study.valuesA[i].toExponential(1):study.valuesA[i].toFixed(2)}</em></i><i class="bar alt" style="--value:${study.valuesB[i]/study.max}"><em>${study.valuesB[i]<.01?study.valuesB[i].toExponential(1):study.valuesB[i].toFixed(2)}</em></i><span>${label}</span></div>`).join("");
    const top=study.max<.01?study.max.toExponential(1):study.max.toFixed(1),mid=study.max<.01?(study.max/2).toExponential(1):(study.max/2).toFixed(1);
    return `<div class="bar-chart"><div class="bar-axis"><span>${top}</span><span>${mid}</span><span>0</span></div><div class="bar-groups">${bars}</div></div>`;
  }
  function setCaseStep(next){
    const study=caseStudies[currentCase];caseStep=(next+study.steps.length)%study.steps.length;
    const model=document.getElementById("case-model");model.dataset.step=String(caseStep);model.querySelector(".model-caption").textContent=study.steps[caseStep];
    document.querySelectorAll("#model-stepper button").forEach((button,i)=>{button.classList.toggle("active",i===caseStep);button.setAttribute("aria-label",`Show model step ${i+1}: ${study.steps[i]}`)});
  }
  function startCaseAnimation(){
    if(caseTimer)root.clearInterval(caseTimer);caseTimer=root.setInterval(()=>setCaseStep(caseStep+1),2700);
    const button=document.getElementById("case-play");button.textContent="Pause animation";button.setAttribute("aria-pressed","false");
  }
  function toggleCaseAnimation(){
    const button=document.getElementById("case-play");
    if(caseTimer){root.clearInterval(caseTimer);caseTimer=null;button.textContent="Play animation";button.setAttribute("aria-pressed","true")}else startCaseAnimation();
  }
  function selectCase(name){
    currentCase=name;caseStep=0;const study=caseStudies[name],experience=document.getElementById("case-experience");experience.dataset.case=name;
    document.querySelectorAll("[data-case].case-choice").forEach(button=>{const active=button.dataset.case===name;button.classList.toggle("active",active);button.setAttribute("aria-selected",String(active));button.tabIndex=active?0:-1});
    document.getElementById("case-kicker").textContent=study.kicker;document.getElementById("case-title").textContent=study.title;document.getElementById("case-context").textContent=study.context;document.getElementById("case-question").textContent=study.question;document.getElementById("result-tag").textContent=study.tag;
    document.getElementById("case-model").innerHTML=caseModelMarkup(name);
    const stepper=document.getElementById("model-stepper");stepper.innerHTML=study.steps.map((_,i)=>`<button type="button" data-model-step="${i}" aria-label="Show model step ${i+1}"></button>`).join("");
    stepper.querySelectorAll("button").forEach(button=>button.addEventListener("click",()=>{setCaseStep(Number(button.dataset.modelStep));if(caseTimer)startCaseAnimation()}));
    document.getElementById("result-chart").innerHTML=resultMarkup(study);
    document.getElementById("result-explanation").innerHTML=`<div class="chart-legend"><span>${study.series[0]}</span><span>${study.series[1]}</span></div>${study.findings.map(([title,text])=>`<div class="finding"><b>${title}</b><p>${text}</p></div>`).join("")}`;
    setCaseStep(0);startCaseAnimation();
  }

  function debounce(fn,delay=140){let timer;return()=>{root.clearTimeout(timer);timer=root.setTimeout(fn,delay)}}
  function init(){
    buildNetwork();drawStoryTrajectories();setStoryStep(0);drawAbcTrajectories();drawAbcPosterior();selectCase("candida");
    document.querySelectorAll("[data-story-step]").forEach(button=>button.addEventListener("click",()=>{stopStory();setStoryStep(Number(button.dataset.storyStep))}));
    document.getElementById("story-play").addEventListener("click",toggleStory);
    const sims=document.getElementById("abc-sims"),accept=document.getElementById("abc-accept");
    sims.addEventListener("input",()=>{document.getElementById("abc-sims-out").textContent=Number(sims.value).toLocaleString();if(!lastAbc)document.getElementById("abc-progress-count").textContent=`0 / ${Number(sims.value).toLocaleString()}`});
    accept.addEventListener("input",()=>{document.getElementById("abc-accept-out").textContent=`${accept.value}%`});
    document.getElementById("abc-run").addEventListener("click",runAbc);document.getElementById("abc-reset").addEventListener("click",resetAbc);
    document.querySelectorAll(".case-choice").forEach(button=>button.addEventListener("click",()=>selectCase(button.dataset.case)));
    document.getElementById("case-play").addEventListener("click",toggleCaseAnimation);
    root.addEventListener("resize",debounce(()=>{drawStoryTrajectories();if(lastAbc){drawAbcTrajectories(lastAbc.samples,lastAbc.accepted);drawAbcPosterior(lastAbc.samples,lastAbc.accepted)}else{drawAbcTrajectories();drawAbcPosterior()}}));
    if(root.matchMedia&&root.matchMedia("(prefers-reduced-motion: reduce)").matches){if(caseTimer)root.clearInterval(caseTimer);caseTimer=null;const button=document.getElementById("case-play");button.textContent="Play animation";button.setAttribute("aria-pressed","true")}
  }

  const api={evolveAllele,observedTrajectory,abcBatch,quantile,caseStudies};
  root.InteractivePoster=api;
  if(typeof module!=="undefined"&&module.exports)module.exports=api;
  if(typeof document!=="undefined"){if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",init);else init()}
})(typeof globalThis!=="undefined"?globalThis:this);
