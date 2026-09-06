(function(root){
  "use strict";

  const Science=root.WorkshopScience;
  const COLORS={blue:"#4e814b",blueDark:"#31583b",blueDeep:"#203628",terra:"#a64135",terraDark:"#7b3048",grid:"#d7d1c7",muted:"#6b6761",paper:"#ffffff"};
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
    titles:["Generate possible worlds","Learn the inverse map","Read the joint posterior"],
    texts:[
      "Draw θ from the prior, run the mechanistic model, and repeat. Each parameter setting produces a possible evolutionary trajectory.",
      "Train a conditional density estimator on paired parameters and simulations. It learns which θ are compatible with a trajectory.",
      "Give the observed data to the trained model. Four KDE levels show which pairs of parameters remain plausible; their tilt reveals a negative trade-off."
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
      kicker:"CANDIDA ALBICANS · ANEUPLOIDY REVERSION",
      title:"Why do some extra chromosomes persist?",
      context:"Naïve trisomic cells were isolated and followed through serial passage. Their trajectories reflect two processes at once: chromosomes are lost, while the resulting genotypes grow at different rates.",
      question:"The same visible trajectory can arise from different combinations of chromosome loss and selection. SBI estimates both together instead of treating disappearance as the mutation rate.",
      takeawayLabel:"What SBI resolves",
      tag:"Collective posterior medians",
      source:null,
      resultType:"candida-rates",
      steps:["Begin with a sorted population enriched for one trisomic haplotype.","Chromosome loss can restore heterozygous disomy at rate μHET.","A second route produces homozygous disomy at rate μLOH; genotype fitness reshapes all three trajectories."],
      rates:[
        {label:"Chr4",abb:.00009824,aab:.00015012},
        {label:"Chr5",abb:.00039690,aab:.00033596},
        {label:"Chr6",abb:.00139748,aab:.00097374},
        {label:"Chr7",abb:.00179746,aab:.00211214}
      ],
      legend:["ABB haplotype","AAB haplotype"],
      findings:[
        [">20× rate span","Chromosome-loss estimates cover more than an order of magnitude across chromosome–haplotype backgrounds."],
        ["Rate is not fate","Chr6 ABB remains common despite a high inferred loss rate because trisomy carries a growth advantage in that background."]
      ]
    },
    ms2:{
      kicker:"MS2 BACTERIOPHAGE · PUBLIC-GOODS DYNAMICS",
      title:"Which viral proteins can rescue neighboring genomes?",
      context:"Low MOI exposes a mutant to its own fitness cost. High MOI frequently places mutant and functional genomes in the same cell, where diffusible products may mask that cost.",
      question:"Low-MOI evolution anchors intrinsic mutation and fitness effects. The high-MOI stage then asks how often each protein’s defect is masked specifically during coinfection.",
      takeawayLabel:"Inference logic",
      tag:"High-MOI MAP + 95% HDI",
      source:{href:"https://www.biorxiv.org/content/10.64898/2026.07.02.736036v1",label:"Read the MS2 preprint ↗"},
      resultType:"ms2-intervals",
      steps:["At MOI 0.1, coinfection is rare and deleterious effects are directly exposed.","At MOI 10, multiple genomes frequently occupy the same host cell.","A functional genome can supply a missing product in trans, allowing a recessive defective genome to persist."],
      intervals:[
        {label:"Maturation",estimate:.240,low:.163,high:.350,control:.190},
        {label:"Coat",estimate:.345,low:.191,high:.389,control:.161},
        {label:"Lysis",estimate:.915,low:.871,high:.994,control:.277},
        {label:"Replicase",estimate:.313,low:.245,high:.338,control:.012}
      ],
      legend:["High-MOI MAP + 95% HDI","Largest low-MOI control upper HDI"],
      findings:[
        ["Lysis stands apart","Its inferred masking probability is about 0.92, consistent with a product shared across the infected cell."],
        ["Sharing is a continuum","Coat and replicase show intermediate signals; maturation has the weakest support relative to controls."]
      ]
    }
  };
  let currentCase="candida",caseStep=0,caseTimer=null;
  function caseModelMarkup(name){
    if(name==="candida")return '<div class="candida-diagram"><span class="candida-state aneu">Tri</span><i class="candida-edge one"></i><small class="candida-rate rate-one">μ<sub>HET</sub></small><span class="candida-state het">HET</span><i class="candida-edge two"></i><small class="candida-rate rate-two">μ<sub>LOH</sub></small><span class="candida-state loh">LOH</span><b class="traveller"></b></div><p class="model-caption"></p>';
    return '<div class="ms2-diagram"><div class="moi-world low"><strong>Low MOI</strong><span class="host-cell"></span><i class="phage-dot"></i></div><div class="moi-world high"><strong>High MOI</strong><span class="host-cell"></span><i class="phage-dot"></i><i class="phage-dot"></i><i class="phage-dot"></i><span class="public-good"></span></div></div><p class="model-caption"></p>';
  }
  function resultMarkup(study){
    if(study.resultType==="candida-rates"){
      const left=130,right=700,width=right-left,top=45,rowGap=62,min=-4.2,max=-2.5;
      const x=value=>left+(Math.log10(value)-min)/(max-min)*width;
      const ticks=[[-4,"10⁻⁴"],[-3.5,"3×10⁻⁴"],[-3,"10⁻³"],[-2.5,"3×10⁻³"]];
      const grid=ticks.map(([value,label])=>{const at=left+(value-min)/(max-min)*width;return `<line x1="${at}" y1="24" x2="${at}" y2="282"/><text x="${at}" y="310" text-anchor="middle">${label}</text>`}).join("");
      const rows=study.rates.map((row,i)=>{const y=top+i*rowGap;return `<text class="row-label" x="108" y="${y+6}" text-anchor="end">${row.label}</text><line class="row-rule" x1="${left}" y1="${y}" x2="${right}" y2="${y}"/><circle class="estimate abb" cx="${x(row.abb)}" cy="${y-8}" r="8"/><circle class="estimate aab" cx="${x(row.aab)}" cy="${y+8}" r="8"/>`}).join("");
      return `<svg class="study-result-svg" viewBox="0 0 760 350" role="img" aria-label="Posterior median chromosome-loss rates for ABB and AAB haplotypes across chromosomes 4 through 7"><g class="result-grid">${grid}${rows}</g><text class="axis-title" x="415" y="342" text-anchor="middle">μHET per generation · logarithmic scale</text></svg>`;
    }
    const left=155,right=700,width=right-left,top=45,rowGap=62;
    const ticks=[0,.5,1].map(value=>{const at=left+width*value;return `<line x1="${at}" y1="24" x2="${at}" y2="282"/><text x="${at}" y="310" text-anchor="middle">${value}</text>`}).join("");
    const rows=study.intervals.map((row,i)=>{const y=top+i*rowGap;return `<text class="row-label" x="133" y="${y+6}" text-anchor="end">${row.label}</text><line class="row-rule" x1="${left}" y1="${y}" x2="${right}" y2="${y}"/><line class="hdi" x1="${left+width*row.low}" y1="${y}" x2="${left+width*row.high}" y2="${y}"/><circle class="estimate" cx="${left+width*row.estimate}" cy="${y}" r="8"/><line class="control-mark" x1="${left+width*row.control}" y1="${y-12}" x2="${left+width*row.control}" y2="${y+12}"/><text class="value-label" x="${Math.min(724,left+width*row.high+11)}" y="${y+5}">${row.estimate.toFixed(2)}</text>`}).join("");
    return `<svg class="study-result-svg" viewBox="0 0 760 350" role="img" aria-label="High-MOI posterior estimates and low-MOI controls for protein-specific masking probability"><g class="result-grid">${ticks}${rows}</g><text class="axis-title" x="425" y="342" text-anchor="middle">probability a deleterious mutation is masked during coinfection</text></svg>`;
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
    document.getElementById("case-kicker").textContent=study.kicker;document.getElementById("case-title").textContent=study.title;document.getElementById("case-context").textContent=study.context;document.getElementById("case-question").textContent=study.question;document.getElementById("case-takeaway-label").textContent=study.takeawayLabel;document.getElementById("result-tag").textContent=study.tag;
    const source=document.getElementById("case-source");if(study.source){source.hidden=false;source.href=study.source.href;source.textContent=study.source.label}else{source.hidden=true}
    document.getElementById("case-model").innerHTML=caseModelMarkup(name);
    const stepper=document.getElementById("model-stepper");stepper.innerHTML=study.steps.map((_,i)=>`<button type="button" data-model-step="${i}" aria-label="Show model step ${i+1}"></button>`).join("");
    stepper.querySelectorAll("button").forEach(button=>button.addEventListener("click",()=>{setCaseStep(Number(button.dataset.modelStep));if(caseTimer)startCaseAnimation()}));
    document.getElementById("result-chart").innerHTML=resultMarkup(study);
    document.getElementById("result-explanation").innerHTML=`<div class="chart-legend"><span>${study.legend[0]}</span><span>${study.legend[1]}</span></div>${study.findings.map(([title,text])=>`<div class="finding"><b>${title}</b><p>${text}</p></div>`).join("")}`;
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
