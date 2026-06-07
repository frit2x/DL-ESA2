// FFNN Regression App (TensorFlow.js)
// Implements data generation, noisy dataset, training of three models

const xsAll = [];
let dataset = null; // {x:[], y:[], xTrain:[], yTrain:[], xTest:[], yTest:[], yTrainNoisy:[], yTestNoisy:[]}
let models = {};

function f(x) {
  return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function randUniform(a, b) { return Math.random() * (b - a) + a; }

function gaussianNoise(std) {
  // Box-Muller
  let u = 0, v = 0;
  while(u === 0) u = Math.random();
  while(v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * std;
}

function generateDataset(N=100, noiseVar=0.05) {
  const xs = [];
  for (let i = 0; i < N; i++) xs.push(randUniform(-2, 2));
  // shuffle
  for (let i = xs.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [xs[i], xs[j]] = [xs[j], xs[i]];
  }
  const ys = xs.map(x => f(x));
  const half = Math.floor(N/2);
  const xTrain = xs.slice(0, half);
  const yTrain = ys.slice(0, half);
  const xTest = xs.slice(half, half*2);
  const yTest = ys.slice(half, half*2);

  const std = Math.sqrt(noiseVar);
  const yTrainNoisy = yTrain.map(y => y + gaussianNoise(std));
  const yTestNoisy = yTest.map(y => y + gaussianNoise(std));

  dataset = {x: xs, y: ys, xTrain, yTrain, xTest, yTest, yTrainNoisy, yTestNoisy};
  plotData();
  setStatus(`Datensatz erzeugt: N=${N}, Rauschen V=${noiseVar.toFixed(3)}`);
}

function buildModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({units:100, activation:'relu', inputShape:[1]}));
  model.add(tf.layers.dense({units:100, activation:'relu'}));
  model.add(tf.layers.dense({units:1, activation:'linear'}));
  const opt = tf.train.adam(0.01);
  model.compile({optimizer: opt, loss: 'meanSquaredError'});
  return model;
}

async function trainModel(name, xTrain, yTrain, epochs=50, batchSize=32, onEpoch=null) {
  if (!dataset) { alert('Bitte zuerst Datensatz erzeugen'); return; }
  models[name] = buildModel();
  const m = models[name];
  const xs = tf.tensor2d(xTrain, [xTrain.length, 1]);
  const ys = tf.tensor2d(yTrain, [yTrain.length, 1]);
  const history = [];
  await m.fit(xs, ys, {
    epochs,
    batchSize,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        history.push(logs.loss);
        if (onEpoch) onEpoch(epoch, logs.loss);
        await tf.nextFrame();
      }
    }
  });
  xs.dispose(); ys.dispose();
  return history;
}

function predictModel(name, xArray) {
  const m = models[name];
  if (!m) return null;
  const t = tf.tensor2d(xArray, [xArray.length,1]);
  const preds = m.predict(t);
  const out = Array.from(preds.dataSync());
  t.dispose(); preds.dispose();
  return out;
}

function mse(a, b) {
  let s=0; for (let i=0;i<a.length;i++){ const d=a[i]-b[i]; s+=d*d;} return s/a.length;
}

function linspace(a,b,n){ const r=[]; for(let i=0;i<n;i++) r.push(a+(b-a)*i/(n-1)); return r; }

/* Plotting functions using Plotly */
function plotData() {
  if (!dataset) return;
  const traceCleanTrain = {x: dataset.xTrain, y: dataset.yTrain, mode:'markers', name:'Train ohne Rauschen', marker:{color:'blue'}};
  const traceCleanTest = {x: dataset.xTest, y: dataset.yTest, mode:'markers', name:'Test ohne Rauschen', marker:{color:'cyan'}};
  const traceNoisyTrain = {x: dataset.xTrain, y: dataset.yTrainNoisy, mode:'markers', name:'Train mit Rauschen', marker:{color:'red'}};
  const traceNoisyTest = {x: dataset.xTest, y: dataset.yTestNoisy, mode:'markers', name:'Test mit Rauschen', marker:{color:'orange'}};
  const layout = {title:'Datensätze', xaxis:{title:'x'}, yaxis:{title:'y'}, height:360};
  Plotly.newPlot('plot-data', [traceCleanTrain, traceCleanTest, traceNoisyTrain, traceNoisyTest], layout, {responsive:true});
}

function plotPrediction(plotId, xPoints, yTrueTrain, yTrueTest, predY, title, lossTrain=null, lossTest=null) {
  const xGrid = linspace(-2,2,300);
  const traceCurve = {x: xGrid, y: predictModelCurve(predY,xGrid), mode:'lines', name:'Model', line:{color:'green'}};
  const traceTrain = {x: dataset.xTrain, y: yTrueTrain, mode:'markers', name:'Train', marker:{color:'blue'}};
  const traceTest = {x: dataset.xTest, y: yTrueTest, mode:'markers', name:'Test', marker:{color:'orange'}};
  const layout = {title, xaxis:{title:'x'}, yaxis:{title:'y'}, height:360};
  Plotly.newPlot(plotId, [traceCurve, traceTrain, traceTest], layout, {responsive:true});
  const lossDiv = document.getElementById('loss-' + plotId.split('-')[1]);
  if (lossDiv) {
    lossDiv.innerText = '';
    if (lossTrain !== null) lossDiv.innerText += `Train MSE: ${lossTrain.toExponential(3)} `;
    if (lossTest !== null) lossDiv.innerText += `| Test MSE: ${lossTest.toExponential(3)}`;
  }
}

// Helper: when we only have predictions at grid points, but here we create a function wrapper
function predictModelCurve(predArray, xGrid) {
  // If predArray is a function, call it; else assume we have model name in predArray
  if (typeof predArray === 'function') return xGrid.map(predArray);
  return predictModel(predArray, xGrid);
}

function setStatus(text) {
  const el = document.getElementById('status');
  if (el) el.textContent = text;
}

async function run() {
  document.getElementById('btn-generate').addEventListener('click', () => {
    const N = parseInt(document.getElementById('numPoints').value);
    const V = parseFloat(document.getElementById('noiseVar').value);
    generateDataset(N, V);
  });

  document.getElementById('btn-train-clean').addEventListener('click', async () => {
    if (!dataset) return alert('Bitte Datensatz erzeugen');
    document.getElementById('discussion-text').innerText = 'Training auf unverrauschten Daten...';
    const epochs = parseInt(document.getElementById('epochsBest').value);
    const h = await trainModel('clean', dataset.xTrain, dataset.yTrain, epochs, 32, (epoch,loss)=> updateLossPlot('clean', epoch, loss));
    // evaluate
    const predTrain = predictModel('clean', dataset.xTrain);
    const predTest = predictModel('clean', dataset.xTest);
    const lossTrain = mse(predTrain, dataset.yTrain);
    const lossTest = mse(predTest, dataset.yTest);
    plotPrediction('plot-clean', null, dataset.yTrain, dataset.yTest, 'clean', 'Vorhersage (ohne Rauschen)', lossTrain, lossTest);
    updateLossChart('clean');
  });

  document.getElementById('btn-train-best').addEventListener('click', async () => {
    if (!dataset) return alert('Bitte Datensatz erzeugen');
    document.getElementById('discussion-text').innerText = 'Training Best-Fit auf verrauschten Daten...';
    const epochs = parseInt(document.getElementById('epochsBest').value);
    const history = await trainModel('best', dataset.xTrain, dataset.yTrainNoisy, epochs, 32, (epoch,loss)=> updateLossPlot('best', epoch, loss));
    const predTrain = predictModel('best', dataset.xTrain);
    const predTest = predictModel('best', dataset.xTest);
    const lossTrain = mse(predTrain, dataset.yTrainNoisy);
    const lossTest = mse(predTest, dataset.yTestNoisy);
    plotPrediction('plot-best', null, dataset.yTrainNoisy, dataset.yTestNoisy, 'best', 'Best-Fit Vorhersage (verrauscht)', lossTrain, lossTest);
    updateLossChart('best');
  });

  document.getElementById('btn-train-over').addEventListener('click', async () => {
    if (!dataset) return alert('Bitte Datensatz erzeugen');
    document.getElementById('discussion-text').innerText = 'Training Overfit auf verrauschten Daten (lange Epochen)...';
    const epochs = parseInt(document.getElementById('epochsOver').value);
    await trainModel('over', dataset.xTrain, dataset.yTrainNoisy, epochs, 32, (epoch,loss)=> updateLossPlot('over', epoch, loss));
    const predTrain = predictModel('over', dataset.xTrain);
    const predTest = predictModel('over', dataset.xTest);
    const lossTrain = mse(predTrain, dataset.yTrainNoisy);
    const lossTest = mse(predTest, dataset.yTestNoisy);
    plotPrediction('plot-over', null, dataset.yTrainNoisy, dataset.yTestNoisy, 'over', 'Overfit Vorhersage (verrauscht)', lossTrain, lossTest);
    updateLossChart('over');
  });

  document.getElementById('btn-save-data').addEventListener('click', () => {
    if (!dataset) return alert('Kein Datensatz zum Speichern');
    const payload = JSON.stringify(dataset);
    const blob = new Blob([payload], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'dataset.json'; a.click(); URL.revokeObjectURL(url);
  });

  document.getElementById('btn-load-data').addEventListener('click', () => {
    const inp = document.createElement('input'); inp.type='file'; inp.accept='application/json';
    inp.onchange = e => {
      const file = e.target.files[0]; if(!file) return;
      const reader = new FileReader();
      reader.onload = () => { dataset = JSON.parse(reader.result); plotData(); };
      reader.readAsText(file);
    };
    inp.click();
  });

  document.getElementById('btn-save-model').addEventListener('click', async () => {
    const name = prompt('Model-Name für IndexedDB (z.B. best):','best'); if(!name) return;
    if (!models[name]) return alert('Kein Modell mit diesem Namen trainiert');
    await models[name].save('indexeddb://' + name);
    alert('Modell gespeichert: ' + name);
  });

  document.getElementById('btn-load-model').addEventListener('click', async () => {
    const name = prompt('Model-Name aus IndexedDB laden','best'); if(!name) return;
    try {
      models[name] = await tf.loadLayersModel('indexeddb://' + name);
      alert('Modell geladen: ' + name);
    } catch(e) { alert('Laden fehlgeschlagen: ' + e.message); }
  });

  // initial dataset
  generateDataset();
}

/* Loss plotting helpers */
const lossHist = {clean:[], best:[], over:[]};
function updateLossPlot(name, epoch, loss) {
  lossHist[name].push(loss);
  // update combined losses chart
  updateLossChart();
}

function updateLossChart() {
  const traces = [];
  for (const k of ['clean','best','over']) {
    if (lossHist[k].length) {
      traces.push({x: Array.from({length: lossHist[k].length}, (_,i)=>i+1), y: lossHist[k], mode:'lines+markers', name: k});
    }
  }
  Plotly.newPlot('plot-losses', traces, {title:'Loss pro Epoche (MSE)', xaxis:{title:'Epoche'}, yaxis:{title:'MSE'}, height:320}, {responsive:true});
}

// Utility to get predictions for a grid using model name
function predictModel(name, xArray) {
  const m = models[name];
  if (!m) return xArray.map(x=>0);
  const t = tf.tensor2d(xArray, [xArray.length,1]);
  const out = m.predict(t).dataSync();
  t.dispose();
  return Array.from(out);
}

// Kick off after DOM is ready
window.addEventListener('DOMContentLoaded', () => {
  run();
  setStatus('Bereit. Klicken Sie auf "Datensatz erzeugen".');
});
