document.addEventListener("DOMContentLoaded", () => {
    // UI Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadPrompt = document.getElementById("upload-prompt");
    const imagePreview = document.getElementById("image-preview");
    
    const classifyBtn = document.getElementById("classify-btn");
    const resetBtn = document.getElementById("reset-btn");
    
    const resultBlock = document.getElementById("result-block");
    const predictionValue = document.getElementById("prediction-value");
    const errorAlert = document.getElementById("error-alert");

    // Probabilities Bar Elements
    const barCat = document.getElementById("prob-bar-cat");
    const barDog = document.getElementById("prob-bar-dog");
    const barWild = document.getElementById("prob-bar-wild");

    const textCat = document.getElementById("prob-val-cat");
    const textDog = document.getElementById("prob-val-dog");
    const textWild = document.getElementById("prob-val-wild");

    let selectedFile = null;

    // Trigger File Picker
    dropZone.addEventListener("click", () => {
        if (!selectedFile) {
            fileInput.click();
        }
    });

    // Demo Images Selection
    const demoCards = document.querySelectorAll(".demo-card");
    demoCards.forEach((card) => {
        card.addEventListener("click", async (e) => {
            e.stopPropagation();
            const fileUrl = card.getAttribute("data-file");
            const fileName = card.getAttribute("data-name");

            try {
                const response = await fetch(fileUrl);
                const blob = await response.blob();
                const file = new File([blob], fileName, { type: blob.type });
                handleFiles([file]);
            } catch (err) {
                showError("Failed to load the selected demo image.");
            }
        });
    });

    // File input selection change
    fileInput.addEventListener("change", (e) => {
        handleFiles(e.target.files);
    });

    // Drag and Drop Events
    ["dragenter", "dragover"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        handleFiles(e.dataTransfer.files);
    });

    // Process Selected File
    function handleFiles(files) {
        if (files.length === 0) return;

        const file = files[0];

        // Check if file is an image
        if (!file.type.startsWith("image/")) {
            showError("The selected file is not a valid image format. Please upload a standard image file.");
            return;
        }

        selectedFile = file;
        clearError();

        // Show image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove("hidden");
            uploadPrompt.classList.add("hidden");
            
            // Enable classify and show reset buttons
            classifyBtn.removeAttribute("disabled");
            resetBtn.classList.remove("hidden");
        };
        reader.readAsDataURL(file);
    }

    // Classify Button Action
    classifyBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        // Set Loading State
        classifyBtn.setAttribute("disabled", "true");
        classifyBtn.textContent = "Classifying...";
        clearError();
        resultBlock.classList.add("hidden");

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "An unexpected error occurred during classification.");
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            showError(error.message);
            classifyBtn.removeAttribute("disabled");
        } finally {
            classifyBtn.textContent = "Classify Image";
            // Keep button disabled until reset, or enable it so they can try again if they want
            classifyBtn.removeAttribute("disabled");
        }
    });

    // Reset Page Action
    resetBtn.addEventListener("click", () => {
        selectedFile = null;
        fileInput.value = "";
        
        imagePreview.src = "";
        imagePreview.classList.add("hidden");
        uploadPrompt.classList.remove("hidden");
        
        classifyBtn.setAttribute("disabled", "true");
        resetBtn.classList.add("hidden");
        resultBlock.classList.add("hidden");
        clearError();
        
        // Reset probabilities
        resetProbabilities();
    });

    // Display Classification Results
    function displayResults(data) {
        // Set Class text
        predictionValue.textContent = `Class: ${data.prediction}`;

        // Extract probabilities
        const catProb = (data.probabilities["Cat"] * 100).toFixed(2);
        const dogProb = (data.probabilities["Dog"] * 100).toFixed(2);
        const wildProb = (data.probabilities["Wild Animal"] * 100).toFixed(2);

        // Update progress bar widths
        barCat.style.width = `${catProb}%`;
        barDog.style.width = `${dogProb}%`;
        barWild.style.width = `${wildProb}%`;

        // Update probability values
        textCat.textContent = `${catProb}%`;
        textDog.textContent = `${dogProb}%`;
        textWild.textContent = `${wildProb}%`;

        // Show result card
        resultBlock.classList.remove("hidden");
        
        // Smooth scroll to results
        resultBlock.scrollIntoView({ behavior: "smooth" });
    }

    function resetProbabilities() {
        barCat.style.width = "0%";
        barDog.style.width = "0%";
        barWild.style.width = "0%";

        textCat.textContent = "0.0%";
        textDog.textContent = "0.0%";
        textWild.textContent = "0.0%";
    }

    // Helper functions for alerts
    function showError(message) {
        errorAlert.textContent = message;
        errorAlert.classList.remove("hidden");
    }

    function clearError() {
        errorAlert.textContent = "";
        errorAlert.classList.add("hidden");
    }
});
