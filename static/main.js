//========================================================================
// Drag and drop image handling
//========================================================================

var fileDrag = document.getElementById("file-drag");
var fileSelect = document.getElementById("file-upload");

// Add event listeners
fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
    // prevent default behaviour
    e.preventDefault();
    e.stopPropagation();

    fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
    // handle file selecting
    var files = e.target.files || e.dataTransfer.files;
    fileDragHover(e);
    for (var i = 0, f; (f = files[i]); i++) {
        previewFile(f);
    }
}

//========================================================================
// Web page elements for functions to use
//========================================================================

var imagePreview = document.getElementById("image-preview"); // the image above (import)
var imageDisplay = document.getElementById("image-display"); // the image below (submit & display result)
var uploadCaption = document.getElementById("upload-caption");
var predResult = document.getElementById("pred-result");
var loader = document.getElementById("loader");

//========================================================================
// Main button events
//========================================================================

function submitGeneralImage() {
    // action for the submit button
    console.log("submit");

    if (!imageDisplay.src || !imageDisplay.src.startsWith("data")) {
        window.alert("Please select an image before submit.");
        return;
    }

    loader.classList.remove("hidden");
    imageDisplay.classList.add("loading");

    // call the predict function of the backend
    predictImage(imageDisplay.src, false);
}

function submitHandwrittenImage() {
    // action for the submit button
    console.log("submit");

    if (!imageDisplay.src || !imageDisplay.src.startsWith("data")) {
        window.alert("Please select an image before submit.");
        return;
    }

    loader.classList.remove("hidden");
    imageDisplay.classList.add("loading");

    // call the predict function of the backend
    predictImage(imageDisplay.src, true);
}

function clearImage() {
    // reset selected files
    fileSelect.value = "";

    // remove image sources and hide them
    imagePreview.src = "";
    imageDisplay.src = "";
    predResult.innerHTML = "";
    document.getElementById("output").value = "综合结果";
    document.getElementById("format").value = "格式化结果";

    hide(imagePreview);
    hide(imageDisplay);
    hide(loader);
    hide(predResult);
    show(uploadCaption);

    imageDisplay.classList.remove("loading");
}

function previewFile(file) {
    // show the preview of the image
    console.log(file.name);
    var fileName = encodeURI(file.name);

    var reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = () => {
        imagePreview.src = URL.createObjectURL(file);

        show(imagePreview);
        hide(uploadCaption);

        // reset
        predResult.innerHTML = "";
        imageDisplay.classList.remove("loading");

        displayImage(reader.result, "image-display");
    };
}

//========================================================================
// Helper functions
//========================================================================

function predictImage(image, is_handwritten) {
    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            'image': image,
            'handwritten': is_handwritten
        })
    })
        .then(resp => {
            if (resp.ok)
                resp.json().then(data => {
                    displayResult(data);
                });
        })
        .catch(err => {
            console.log("An error occured", err.message);
            window.alert("Oops! Something went wrong.");
        });
}

function displayImage(image, id) {
    // display image on given id <img> element
    let display = document.getElementById(id);
    display.src = image;
    show(display);
}

function displayResult(data) {
    // display the result
    // imageDisplay.classList.remove("loading");
    hide(loader);
    // predResult.innerHTML = data.result;
    // show(predResult);
    imageDisplay.classList.remove("loading")
    document.getElementById("output").value = data.res_dict['output'];
    document.getElementById("format").value = JSON.stringify(data.res_dict['format'], null, 2);
    displayImage(data.res_image, "image-display");
}

function hide(el) {
    // hide an element
    el.classList.add("hidden");
}

function show(el) {
    // show an element
    el.classList.remove("hidden");
}