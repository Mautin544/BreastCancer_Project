// ------------------------
// Sample Autofill Function
// ------------------------
function fillSample() {
    const sample = [
        14.2, 20.1, 90.2, 600,           // Mean Radius → Mean Area
        0.1, 0.13, 0.12, 0.08, 0.18,     // Mean Smoothness → Mean Symmetry
        0.06, 0.05, 0.05, 0.02, 0.03,    // Mean Fractal Dim → Area Error
        0.1, 0.12, 0.11, 0.08, 0.18,     // Smoothness Error → Symmetry Error
        16, 25, 105, 700, 0.12,          // Fractal Dim Error → Worst Area
        0.15, 0.14, 0.1, 0.2,            // Worst Smoothness → Worst Concave Points
        0.18, 0.07                        // Worst Symmetry, Worst Fractal Dimension
    ];

    document.querySelectorAll("input").forEach((el, i) => {
        if (sample[i] !== undefined) {
            el.value = sample[i];
        }
    });
}

// ------------------------
// Animated Confidence Bar
// ------------------------
function animateConfidence(probability, predictionClass) {
    const resultDiv = document.querySelector(".result");
    if (!resultDiv) return;

    let barContainer = resultDiv.querySelector(".confidence-bar-container");

    if (!barContainer) {
        barContainer = document.createElement("div");
        barContainer.className = "confidence-bar-container";
        resultDiv.appendChild(barContainer);
    }

    // Clear previous bar
    barContainer.innerHTML = "";

    const bar = document.createElement("div");
    bar.className = "confidence-bar";

    // Color-code based on prediction
    if (predictionClass === "benign") {
        bar.style.background = "linear-gradient(135deg, #00c853, #00e676)";
    } else if (predictionClass === "malignant") {
        bar.style.background = "linear-gradient(135deg, #ff1744, #ff5252)";
    } else {
        bar.style.background = "linear-gradient(135deg, #00c6ff, #0072ff)";
    }

    barContainer.appendChild(bar);

    // Animate width
    bar.style.width = "0%";
    setTimeout(() => {
        bar.style.width = (probability * 100).toFixed(2) + "%";
    }, 100);
}

// ------------------------
// Initialize Autofill Button & Confidence Bar
// ------------------------
window.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("form");

    // Check if autofill button exists, otherwise create it
    let autofillBtn = document.querySelector(".autofill-btn");
    if (!autofillBtn) {
        autofillBtn = document.createElement("button");
        autofillBtn.type = "button";
        autofillBtn.textContent = "🔹 Autofill Sample Data";
        autofillBtn.className = "autofill-btn";
        form.insertBefore(autofillBtn, form.firstChild);
    }

    autofillBtn.addEventListener("click", fillSample);

    // Animate confidence bar if POST result exists
    const confidenceElem = document.querySelector(".confidence strong");
    const resultDiv = document.querySelector(".result");

    if (confidenceElem && resultDiv) {
        const prob = parseFloat(confidenceElem.textContent);
        const predictionClass = resultDiv.classList.contains("benign")
            ? "benign"
            : resultDiv.classList.contains("malignant")
            ? "malignant"
            : "";
        animateConfidence(prob, predictionClass);
    }
});
