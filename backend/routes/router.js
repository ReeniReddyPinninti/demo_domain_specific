// const express = require("express");
// const router = express.Router();
// const pdfParse = require("pdf-parse");
// const fileUpload = require("express-fileupload");
// const controllers = require("../controllers/userControllers");
// const datacon = require("../controllers/dataContoller");
// const FormData = require("form-data");

// // Middleware setup
// router.use(fileUpload());

// router.post("/signin" , controllers.userLogin);
// router.post("/signup", controllers.userregister);
// router.post("/otp", controllers.userOtpSend);
// //router.post("/savedata", datacon.UserData);
// //router.post("/sendData", datacon.getData);
// router.post("/getRes", datacon.query);
// //router.post("/saveAnswer", datacon.saveAns);
// //router.post("/getAnswers",datacon.getAns);


// router.post("/savedata", (req, res) => {
//     const username = req.body.username;  // Get username from request body
//     const pdfPath = req.files ? req.files.pdfFile : null;  // Assuming the PDF is being uploaded via `req.files`
  
//     if (!username || !pdfPath) {
//       return res.status(400).json({ error: "Username and PDF file are required." });
//     }
  
//     // Create a FormData instance
//     const form = new FormData();
//     form.append("username", username);
//     form.append("pdf_file", fs.createReadStream(pdfPath.tempFilePath));  // Get the path of the uploaded file
  
//     // Send the request to FastAPI
//     axios.post("http://0.0.0.0:5100/embeddings", form, {
//       headers: {
//         ...form.getHeaders(),  // Include the necessary headers for FormData
//       },
//     })
//       .then(response => {
//         // Return the FastAPI response back to the client
//         res.json(response.data);
//       })
//       .catch(error => {
//         console.error("Error:", error.response ? error.response.data : error.message);
//         res.status(500).json({ error: "Error sending data to FastAPI." });
//       });
//   });

const express = require("express");
const router = express.Router();
const fs = require("fs");
const axios = require("axios");
const fileUpload = require("express-fileupload");
const FormData = require("form-data");
const controllers = require("../controllers/userControllers");
const datacon = require("../controllers/dataContoller");

// Middleware setup
router.use(fileUpload());

router.post("/signin", controllers.userLogin);
router.post("/signup", controllers.userregister);
router.post("/otp", controllers.userOtpSend);
// router.post("/savedata", datacon.UserData);
// router.post("/sendData", datacon.getData);
router.post("/getRes", datacon.query);
// router.post("/saveAnswer", datacon.saveAns);
// router.post("/getAnswers", datacon.getAns);

router.post("/savedata", (req, res) => {
  const username = req.body.username; // Get username from request body
  const pdfPath = req.files ? req.files.pdfFile : null; // Assuming the PDF is being uploaded via `req.files`

  if (!username || !pdfPath) {
    return res.status(400).json({ error: "Username and PDF file are required." });
  }

  // Create a FormData instance
  const form = new FormData();
  form.append("username", username);
  form.append("pdf_file", pdfPath.data, { filename: pdfPath.name }); // Using the `data` buffer directly

  // Send the request to FastAPI
  axios
    .post("http://0.0.0.0:5100/embeddings", form, {
      headers: {
        ...form.getHeaders(), // Include the necessary headers for FormData
      },
    })
    .then((response) => {
      // Return the FastAPI response back to the client
      res.json(response.data);
    })
    .catch((error) => {
      console.error("Error:", error.response ? error.response.data : error.message);
      res.status(500).json({ error: "Error sending data to FastAPI." });
    });
});

  

// POST route to extract text from PDF
router.post("/extract-text", (req, res) => {
    if (!req.files || !req.files.pdfFile) {
        return res.status(400).send("No PDF file uploaded.");
    }

    const pdfFile = req.files.pdfFile;

    pdfParse(pdfFile.data).then(parsedData => {
        res.send(parsedData.text);
    }).catch(err => {
        console.error("Error parsing PDF:", err);
        res.status(500).send("Error parsing PDF.");
    });
});

router.post("/addLink", (req,res) => {
    const {link} = req.body;
    if (!link)
    {
        return res.status(400).send("No link uploaded");
    }
    else
    {
        return res.status(200).send("Link uploaded");
    }
})
module.exports = router;