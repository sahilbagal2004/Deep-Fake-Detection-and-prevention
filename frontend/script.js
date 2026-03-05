async function upload() {

const fileInput = document.getElementById("file")
const file = fileInput.files[0]

if(!file){
alert("Please select an image first")
return
}

document.getElementById("result").innerText = "Processing..."

let formData = new FormData()
formData.append("file", file)

try{

let response = await fetch("http://127.0.0.1:5000/predict",{
method:"POST",
body:formData
})

let data = await response.json()

document.getElementById("result").innerText =
"Prediction: " + data.result

}catch(error){

document.getElementById("result").innerText =
"Error connecting to server"

}

}