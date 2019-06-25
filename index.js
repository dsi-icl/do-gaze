const express = require('express');
const port = 3000;
const app = express();

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.get('/gaze', function (req, res) {
    console.log('Hello,');
    res.json({
        message: "World !"
    });
})
app.post('/gaze', function (req, res) {
    console.log(req.body);
    res.json(req.body);
})

app.listen(port, function () {
    console.log("Server is running on " + port + " port");
});
