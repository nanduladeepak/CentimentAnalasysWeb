document.getElementById('upload-form').addEventListener('submit', function (e) {
    e.preventDefault();

    var formData = new FormData(this);

    fetch('/upload', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Video uploaded successfully!');
            chart.style.display = 'block'
            vedioUpload.style.display = 'None'

            plotBarChart(data.sentiment);
        } else {
            alert('Error uploading video.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
});