async function getPredictions() {
    const crop = document.getElementById('crop').value;
    const N = document.getElementById('N').value;
    const P = document.getElementById('P').value;
    const K = document.getElementById('K').value;
    const ph = document.getElementById('ph').value;

    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            label_encoded: crop,
            N: N,
            P: P,
            K: K,
            ph: ph
        })
    });

    const data = await response.json();
    document.getElementById('predicted_N').innerText = data.predicted_N;
    document.getElementById('predicted_P').innerText = data.predicted_P;
    document.getElementById('predicted_K').innerText = data.predicted_K;
    document.getElementById('predicted_pH').innerText = data.predicted_pH;
}
