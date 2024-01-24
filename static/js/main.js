chart = document.querySelector('div#chart-container')
vedioUpload = document.querySelector('div#upload-vedio')
backButton = document.querySelector('div#chart-container > button#back')



function back(){
    chart.style.display = 'None'
    vedioUpload.style.display = 'block'
}

back()

backButton.addEventListener('click',(e)=>{back()})