def MasterDset(n_imgs, n_test, linklist, labelnames):
    data1 = np.empty_like(np.zeros([1,784]))
    data2 = np.empty_like(np.zeros([1,784]))
    for link in linklist:
        print("Loading",link)
        response = requests.get(link)
        response.raise_for_status()
        #temp = np.load(io.BytesIO(response.content))[:n_imgs]
        temp = np.random.permutation(np.load(io.BytesIO(response.content)))
        temp1 = temp[:n_imgs]
        temp2 = temp[n_imgs:n_imgs+n_test]
        #print(np.min(temp), np.max(temp))
        data1 = np.concatenate((data1, temp1), axis=0)
        data2 = np.concatenate((data2, temp2), axis=0)
        #print(data1.shape, data2.shape)
    data1 = data1[1:,None,:]/255
    data2 = data2[1:,None,:]/255
    data1[data1>0.5] = 1
    data1[data1<=0.5] = 0
    data2[data2>0.5] = 1
    data2[data2<=0.5] = 0
    return Dset(data1, labelnames), Dset(data2, labelnames)

class Dset(Dataset):
    def __init__(self, data, labelnames):
        self.labelnames = labelnames
        self.data = data
        self.label = torch.outer(torch.arange(0, 13), torch.ones(data.shape[0]//13)).flatten()
        
    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float().reshape(1,28,28), self.label[index]
    
    def __len__(self):
        return len(self.label)
      
def make_prediction_canvas(canvas_size, prediction_func_name):
        canvas_html = '''
<div>
<p>Drawing canvas:</p>
<canvas id="canvas" width="''' + str(canvas_size[0]) + '''" height="''' + str(canvas_size[1]) + '''" style="border: 5px solid black"></canvas>
<button onclick="predict()">Predict</button>
<button onclick="clear_canvas()">Clear canvas</button>
<p id="predictionfield">Prediction:</p>
</div>
<script type="text/Javascript">
function prediction_callback(data){
    if (data.msg_type === 'execute_result') {
        document.getElementById("predictionfield").innerHTML = "Prediction: " + data.content.data['text/plain']
        /*$('#predictionfield').html("Prediction: " + data.content.data['text/plain'])*/
    } else {
        console.log(data)
    }
}
function predict(){
    var imgData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    imgData = Array.prototype.slice.call(imgData.data).filter(function (data, idx) { return idx % 4 == 3; })
    var kernelAPI = undefined
    try {
      //check if defined
      if (IPython) {
        kernelAPI = "IPython"
      }
    } catch(err) {
    }
    try {
      //check if defined
      if (google) {
        kernelAPI = "google"
      }
    } catch(err) {
    }
    if (kernelAPI === "IPython") {
        var command = "''' + prediction_func_name + '''(" + JSON.stringify(imgData) + ")"
        document.getElementById("predictionfield").innerHTML = "Prediction: calculating..."
        /*$('#predictionfield').html("Prediction: calculating...")*/
        var kernel = IPython.notebook.kernel;
        kernel.execute(command, {iopub: {output: prediction_callback}}, {silent: false});
    } else if (kernelAPI === "google") {
        google.colab.kernel.invokeFunction("''' + prediction_func_name + '''", [imgData], {})
        .then(function(result) {
            prediction_callback({msg_type: 'execute_result', content: {data: result.data}})
        })
    } else {
        console.error('no kernel api found to invoke predictions!')
    }
}
canvas = document.getElementById('canvas')
ctx = canvas.getContext("2d")
var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();
var paint;
function clear_canvas() {    
    clickX = new Array();
    clickY = new Array();
    clickDrag = new Array();
    
    redraw();
    document.getElementById("predictionfield").innerHTML = "Prediction: ";
}
function addClick(x, y, dragging)
{
  clickX.push(x);
  clickY.push(y);
  clickDrag.push(dragging);
}
var canvas = document.getElementById("canvas")
/*$('#canvas').mousedown(*/
canvas.addEventListener('mousedown', function(e){
  var boundingRect = canvas.getBoundingClientRect()
  var mouseX = e.pageX - boundingRect.left;
  var mouseY = e.pageY - boundingRect.top;
  
  paint = true;
  addClick(mouseX, mouseY);
  redraw();
});
/*$('#canvas').mousemove(*/
canvas.addEventListener('mousemove', function(e){
  if(paint){
    var boundingRect = canvas.getBoundingClientRect()
    addClick(e.pageX - boundingRect.left, e.pageY - boundingRect.top, true);
    redraw();
  }
});
/*$('#canvas').mouseup(*/
canvas.addEventListener('mouseup', function(e){
  paint = false;
});
/*$('#canvas').mouseleave(*/
canvas.addEventListener('mouseleave', function(e){
  paint = false;
});
function redraw(){
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // Clears the canvas
  
  ctx.strokeStyle = '#000000';//"#df4b26";
  ctx.lineJoin = "round";
  ctx.lineWidth = 10;
  for(var i=0; i < clickX.length; i++) {
    ctx.beginPath();
    if(clickDrag[i] && i){
      ctx.moveTo(clickX[i-1], clickY[i-1]);
     }else{
       ctx.moveTo(clickX[i]-1, clickY[i]);
     }
     ctx.lineTo(clickX[i], clickY[i]);
     ctx.closePath();
     ctx.stroke();
  }
}
</script>
'''
        display(HTML(canvas_html)) 

def predict_func(img_data):
    #transform the list into a 2d array
    img_data = np.asarray(img_data, dtype=np.float32).reshape(*canvas_size)
    #transform the numpy array into a torch tensor
    x = torch.tensor(img_data, dtype=torch.float).to(device)
    x /= 255
    #move it to the GPU is possible
    #scale it down from 280x280 to 28x28 pixels using bilinear interpolation
    x = torch.nn.functional.interpolate(x.unsqueeze_(dim=0).unsqueeze_(dim=1), size=(28, 28), mode='bilinear', align_corners=True).squeeze_(dim=1).squeeze_(dim=0)
    #flip dimensions to match dataset
    #x = x.transpose(0, 1)
    #add a batch dimension with batch_size=1
    x.unsqueeze_(dim=0)
    #add a color dimension with color channel count=1
    x.unsqueeze_(dim=1)
    x[x>0.5] = 1
    x[x<=0.5] = 0
    #plt.figure()
    #plt.imshow(x.squeeze().detach().cpu().numpy())
    #plt.show()
    #print(torch.min(x), torch.max(x))
    #switch model to evaluation mode (only necessary for some modules like dropout and batch normalization, but better to always have it rather than forget it when needed)
    model.eval()
    #predict the label for the input
    global out
    with torch.no_grad():#we don't want to store information for gradient computation
        out = model(x)
    #get the most likely label
    #print(F.softmax(out, dim=-1))
    pred_label = out.argmax(1)
    #print(pred_label)
    pred_label = pred_label.item()
    #Image.fromarry(img_data).save("images/"+str(time())+"_"+str(pred_label)+".png")
    #return the predicted class name to the HTML-framework (to be displayed below)
    return DS_test.labelnames[pred_label]
