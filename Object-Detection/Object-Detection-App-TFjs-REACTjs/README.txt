
git clone https://github.com/nicknochnack/ReactComputerVisionTemplate.git (1.6MB)

npm install (332+MB - node_modules)

ReactComputerVisionTemplate>src>App.js
//1. TODO - Import required model here
import * as cocossd from "@tensorflow-models/coco-ssd";

//3. TODO - Load network 
const net = await cocossd.load();

//4. TODO - Make Detections
const obj = await net.detect(video);
console.log(obj);

On the package.json, under the "scripts", put the following:
"start": "react-scripts --openssl-legacy-provider start",

On your browser, choose Safari>Settings>Advanced and on the bottom, tick the check box "Show features for web developers"

npm start

Access on your browser: 
http://localhost:3000

Develop>Show JavaScript Console and you can see a line like this:
[Log] [{bbox: [1.8127632141113281, 0.5729770660400391, 636.0354423522949, 477.8588676452637], class: "person", score: 0.8662505149841309}] (1) (main.chunk.js, line 152)

On ReactComputerVisionTemplate>src create utilities.js and put the following:

export const drawRect = (detections, ctx) =>{
    detections.forEach(prediction=>{
        // Get Prediction Results
        const [x,y,width,height] = prediction['bbox'];
        const text = prediction['class'];

        // Set Styling
        const color = 'green'
        // const color = '#' + Math.floor(Math.random()*16777215).toString(16);
        ctx.strokeStyle = color
        ctx.font = '18px Arial'
        ctx.fillStyle = color

        // Draw retangles and text
        ctx.beginPath()
        ctx.fillText(text, x, y)
        ctx.rect(x, y, width, height)
        ctx.stroke()
    })
}

//2. TODO - Import drawing utility here
import { drawRect } from "./utilities";

//5. TODO - Update drawing utility
drawRect(obj, ctx);

Watch: https://www.youtube.com/watch?v=uTdUUpfA83s&list=PLgNJO2hghbmhUeJuv7PyVYgzhlgt2TcSr&index=5
Watch: https://youtu.be/uTdUUpfA83s?si=aL8BcEQDJdW8__Su


