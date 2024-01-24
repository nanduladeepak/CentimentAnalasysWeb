var sentimentLabels = ['neutral','calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
function plotBarChart(sentiment) {
    data = [...sentiment]
    data.unshift(new Array(8).fill(0))
    // Labels for the x-axis (assuming equal number of data points for each line)
    var data = data[0].map((_, colIndex) => data.map(row => row[colIndex]));
    var labels = Array.from({ length: data[0].length }, (_, i) => i + 1);

    // Generate random colors for each line
    var colors = Array.from({ length: data.length }, () => getRandomColor());

    // Create a data object for Chart.js
    var chartData = {
        labels: labels,
        datasets: data.map((line, index) => ({
            label: sentimentLabels[index],
            data: line,
            borderColor: colors[index],
            fill: false
        }))
    };

    // Create the line chart
    var ctx = document.getElementById('lineChart').getContext('2d');
    var myLineChart = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            scales: {
                xAxes: [{
                    type: 'linear',
                    position: 'bottom'
                }],
                yAxes: [{
                    type: 'linear',
                    position: 'left'
                }]
            }
        }
    });
    console.log(sentiment)
}

function getRandomColor() {
    var letters = '0123456789ABCDEF';
    var color = '#';
    for (var i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}