const rawDataset = [
["SHOCKING: Doctors HATE this one weird trick that cures cancer overnight!!!","fake"],
["You won't believe what this celebrity did to lose 40 pounds in 3 days","fake"],
["BREAKING: Government secretly hiding cure for baldness, insiders reveal","fake"],
["Scientists CONFIRM the earth is actually flat, cover-up exposed","fake"],
["Miracle pill melts belly fat while you sleep, doctors are furious","fake"],
["Local mom discovers ancient secret that big pharma doesn't want you to know","fake"],
["Aliens spotted landing near small town, government denies everything","fake"],
["This ONE fruit can reverse aging, scientists baffled and terrified","fake"],
["Banned vaccine ingredient found to control your mind, leaked memo shows","fake"],
["Billionaire admits the moon landing was completely faked in secret studio","fake"],
["Drink this every morning and never see a doctor again, they hate her","fake"],
["Shocking video proves 5G towers are spying on your dreams","fake"],
["Secret society controls world weather using hidden machines, whistleblower says","fake"],
["You'll never guess what happens when you mix these two household items","fake"],
["Celebrity secretly replaced by clone, insiders swear it's true","fake"],
["Doctors baffled as woman cures diabetes with this kitchen spice","fake"],
["Government bans this common vegetable, refuses to explain why","fake"],
["Man discovers time travel in his garage, physicists refuse to comment","fake"],
["This shocking photo will change how you see gravity forever","fake"],
["Secret ingredient in soda is turning people into zombies, experts warn","fake"],
["Politician caught on tape admitting elections are completely fake","fake"],
["New study PROVES chocolate cures every disease known to man","fake"],
["Hidden camera catches Bigfoot working at local grocery store","fake"],
["Experts SHOCKED as this app reads your mind through your phone camera","fake"],
["Ancient prophecy predicted this exact headline, numerologists claim","fake"],
["Ban this now! Everyday spice found to cause instant memory loss","fake"],
["Leaked documents reveal moon is actually a hologram projected by NASA","fake"],
["This common household chemical is secretly poisoning your children, doctors hide truth","fake"],
["You won't believe how this teenager became a billionaire overnight using one trick","fake"],
["Miracle water cures blindness in seconds, hospitals refuse to sell it","fake"],
["Secret government files reveal dinosaurs are still alive in hidden valley","fake"],
["Doctors terrified as new fruit diet reverses death, patients report","fake"],
["This app can predict your death date with 100 percent accuracy, users claim","fake"],
["Shocking: Your microwave is secretly recording everything you say","fake"],
["Scientists refuse to reveal why cats can see into other dimensions","fake"],
["Wake up! Mainstream media hiding proof that gravity is optional","fake"],
["This one household plant absorbs all your stress instantly, experts stunned","fake"],
["Local woman turns water into gold using kitchen blender, refuses to share method","fake"],
["Banned exercise burns fat 10 times faster, gyms don't want you to know","fake"],
["Secret formula found in ancient scroll cures aging instantly, historians shocked","fake"],
["City council approves $4.2 million budget for downtown road repairs next fiscal year","real"],
["Researchers at the university published findings on battery efficiency in a peer-reviewed journal","real"],
["The central bank left interest rates unchanged, citing steady inflation data","real"],
["Local school district announces new bus routes ahead of the fall semester","real"],
["A study published in a medical journal found a modest link between sleep and memory retention","real"],
["The city's water utility completed scheduled maintenance on the treatment plant this week","real"],
["Officials confirmed the bridge repair project will be completed by early next year","real"],
["The company reported quarterly earnings that were roughly in line with analyst expectations","real"],
["A new public transit line is set to open next spring after years of construction","real"],
["Weather officials say a cold front will bring lower temperatures to the region this weekend","real"],
["The hospital opened a new wing dedicated to pediatric care after a two-year renovation","real"],
["State lawmakers passed a bill increasing funding for rural broadband access","real"],
["The nonprofit organization released its annual report detailing program outcomes","real"],
["Election officials said voter turnout was higher than in the previous midterm cycle","real"],
["Researchers presented preliminary results at the conference and called for further study","real"],
["The airline announced it will add two new routes starting in the fourth quarter","real"],
["A recall was issued for a batch of products due to a labeling error, the company said","real"],
["The museum will host a new exhibit on regional history beginning next month","real"],
["Public health officials recommended updated guidance based on recent surveillance data","real"],
["The city council voted to approve funding for a new public library branch","real"],
["A federal agency released updated safety guidelines for consumer electronics","real"],
["The university's engineering department received a grant to study renewable energy storage","real"],
["Local farmers reported a slightly below average harvest due to a dry summer","real"],
["The town's fire department responded to a small warehouse fire with no injuries reported","real"],
["A new report from the labor department showed unemployment held steady last month","real"],
["The company said it will invest in expanding its manufacturing facility over the next two years","real"],
["Health officials urged residents to get seasonal flu vaccinations ahead of winter","real"],
["Transportation officials began a study on reducing traffic congestion downtown","real"],
["The school board approved a revised curriculum after months of community feedback","real"],
["Researchers found that regular exercise was associated with improved cardiovascular health in the study","real"],
["The utility company advised customers of a planned power outage for scheduled maintenance","real"],
["A panel of experts testified before the committee about infrastructure funding priorities","real"],
["The city's planning commission approved a proposal for a mixed use development downtown","real"],
["Officials said the investigation into the incident is ongoing and declined further comment","real"],
["The observatory announced the discovery of a distant exoplanet using new telescope data","real"],
["A community survey found most residents support the proposed park renovation","real"],
["The agriculture department released its seasonal crop forecast for the coming year","real"],
["Local officials announced road closures ahead of the annual marathon this weekend","real"],
["The company's board of directors approved a new chief financial officer following a search","real"],
["Researchers cautioned that more data is needed before drawing firm conclusions from the trial","real"]
];

const STOPWORDS = new Set(["the","a","an","of","in","on","at","to","for","and","or","is","are","was","were","be","been","this","that","it","its","as","with","by","from","has","have","had","will","said","says"]);

function tokenize(text){
  return text.toLowerCase()
    .replace(/[^a-z0-9'\s]/g," ")
    .split(/\s+/)
    .filter(w => w.length > 1 && !STOPWORDS.has(w));
}

function trainNaiveBayes(dataset){
  const classes = ["fake","real"];
  const wordCounts = {fake:{}, real:{}};
  const classTotalWords = {fake:0, real:0};
  const classDocs = {fake:0, real:0};
  const vocab = new Set();

  dataset.forEach(([text,label]) => {
    classDocs[label]++;
    tokenize(text).forEach(w => {
      vocab.add(w);
      wordCounts[label][w] = (wordCounts[label][w]||0) + 1;
      classTotalWords[label]++;
    });
  });

  const totalDocs = dataset.length;
  const priors = {
    fake: Math.log(classDocs.fake/totalDocs),
    real: Math.log(classDocs.real/totalDocs)
  };
  const V = vocab.size;

  function wordLogProb(word, label){
    const count = wordCounts[label][word] || 0;
    return Math.log((count + 1) / (classTotalWords[label] + V));
  }

  return {priors, wordLogProb, vocab, classTotalWords, V, wordCounts};
}

const model = trainNaiveBayes(rawDataset);

function predict(text){
  const tokens = tokenize(text);
  let logFake = model.priors.fake;
  let logReal = model.priors.real;
  const contributions = [];

  tokens.forEach(t => {
    const lf = model.wordLogProb(t,"fake");
    const lr = model.wordLogProb(t,"real");
    logFake += lf;
    logReal += lr;
    if(model.vocab.has(t)){
      contributions.push({word:t, diff: lf - lr});
    }
  });

  const maxLog = Math.max(logFake, logReal);
  const eFake = Math.exp(logFake - maxLog);
  const eReal = Math.exp(logReal - maxLog);
  const pFake = eFake / (eFake + eReal);
  const pReal = 1 - pFake;

  return {pFake, pReal, contributions, verdict: pFake > pReal ? "fake" : "real"};
}

function renderEvidence(originalText, contributions){
  const contribMap = {};
  contributions.forEach(c => {
    if(!(c.word in contribMap) || Math.abs(c.diff) > Math.abs(contribMap[c.word])){
      contribMap[c.word] = c.diff;
    }
  });

  const words = originalText.split(/(\s+)/);
  return words.map(w => {
    if(/^\s+$/.test(w) || w.length === 0) return w;
    const clean = w.toLowerCase().replace(/[^a-z0-9']/g,"");
    const diff = contribMap[clean];
    if(diff === undefined || Math.abs(diff) < 0.15) return `<span class="tok">${escapeHtml(w)}</span>`;
    const cls = diff > 0 ? "lean-fake" : "lean-real";
    return `<span class="tok ${cls}">${escapeHtml(w)}</span>`;
  }).join("");
}

function escapeHtml(s){
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function topWords(label, n){
  const entries = Object.entries(model.wordCounts[label]);
  entries.sort((a,b) => b[1]-a[1]);
  return entries.slice(0,n);
}

function runAnalysis(){
  const text = document.getElementById("input-text").value.trim();
  const report = document.getElementById("report");
  if(!text){
    report.innerHTML = '<div class="empty-report">Enter some text to analyze.</div>';
    return;
  }

  const result = predict(text);
  const pct = label => Math.round((label==="fake"?result.pFake:result.pReal)*100);

  const contributions = result.contributions.slice().sort((a,b)=>Math.abs(b.diff)-Math.abs(a.diff));
  const topFakeSignals = contributions.filter(c=>c.diff>0.15).slice(0,5);
  const topRealSignals = contributions.filter(c=>c.diff<-0.15).slice(0,5);

  report.innerHTML = `
    <div class="stamp ${result.verdict}">${result.verdict === "fake" ? "Likely fake" : "Likely real"}</div>
    <div class="conf-row"><span>Fake ${pct("fake")}%</span><span>Real ${pct("real")}%</span></div>
    <div class="bar"><div class="bar-fake" style="width:${pct("fake")}%"></div><div class="bar-real" style="width:${pct("real")}%"></div></div>

    <div class="legend">
      <span><span class="dot" style="background:var(--amber)"></span>Leans fake</span>
      <span><span class="dot" style="background:var(--teal)"></span>Leans real</span>
    </div>

    <div class="evidence-box">${renderEvidence(text, contributions)}</div>

    <div class="top-words">
      <div>
        <h3 class="fake">Top fake-leaning signals</h3>
        <ul>${topFakeSignals.length ? topFakeSignals.map(c=>`<li>${escapeHtml(c.word)} <span>+${c.diff.toFixed(2)}</span></li>`).join("") : "<li>None detected</li>"}</ul>
      </div>
      <div>
        <h3 class="real">Top real-leaning signals</h3>
        <ul>${topRealSignals.length ? topRealSignals.map(c=>`<li>${escapeHtml(c.word)} <span>${c.diff.toFixed(2)}</span></li>`).join("") : "<li>None detected</li>"}</ul>
      </div>
    </div>
  `;
}

document.getElementById("run-btn").addEventListener("click", runAnalysis);

const samples = [
  "SHOCKING: This one weird trick doctors don't want you to know melts fat overnight!!!",
  "The city council approved a $2 million budget increase for park maintenance next year.",
  "Secret government files reveal the moon landing was staged in a hidden studio.",
  "Researchers at the university published a peer-reviewed study on renewable energy storage."
];
const samplesEl = document.getElementById("samples");
samples.forEach((s,i) => {
  const btn = document.createElement("button");
  btn.className = "sample-btn";
  btn.textContent = "Sample " + (i+1);
  btn.addEventListener("click", () => {
    document.getElementById("input-text").value = s;
  });
  samplesEl.appendChild(btn);
});

runAnalysis();