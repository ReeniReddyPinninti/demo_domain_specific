// // const express = require("express")
// // const app = express()
// // const routesHandler= require("./routes/router.js");
// // const cors = require("cors")
// // const path =require("path")
// // const fileUpload = require("express-fileupload");

// // //const route

// // app.use(cors())
// // app.use(fileUpload());
// // app.use("/", routesHandler);
// // const PORT= 4000;
// // app.listen(PORT,()=>{
// //     console.log(`Running on ${PORT}`)
// // })
// require("dotenv").config();
// const express = require("express");
// const app = express();
// const routesHandler = require("./routes/router.js");
// const cors = require("cors");
// const fileUpload = require("express-fileupload");
// require("./schema/conn.js");
// const path  = require('path');


// // Middleware setup
// app.use(express.json());
// app.use(cors());
// app.use(fileUpload());

// // Routes setup
// app.use(routesHandler);

// const multer = require("multer");

// // Use memoryStorage to keep the file in memory instead of saving it to the disk
// const storage = multer.memoryStorage();
// const upload = multer({ storage: storage });

// app.post("/upload", upload.single("pdfFile"), (req, res) => {
//     if (!req.file) {
//         return res.status(400).json({ error: "No file uploaded." });
//     }

//     // Access file data in memory (req.file.buffer)
//     const fileData = req.file.buffer;

//     console.log("Uploaded file:", req.file);  // Log file info
//     console.log("File data (buffer):", fileData);  // Log the file data (binary buffer)

//     // Process the file here (for example, convert, analyze, or send elsewhere)

//     res.send("File uploaded successfully without saving.");
// });

// // Start the server
// const PORT = 4000;
// app.listen(PORT, () => {
//     console.log(`Server is running on http://localhost:${PORT}`);
// });

require("dotenv").config();
const express = require("express");
const app = express();
const routesHandler = require("./routes/router.js");
const cors = require("cors");
require("./schema/conn.js");
const path = require("path");

// Multer for file upload handling
const multer = require("multer");

// Use memoryStorage to keep the file in memory instead of saving it to disk
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Middleware setup
app.use(express.json());
app.use(cors());

// Routes setup
app.use(routesHandler);

// Handle the file upload route
app.post("/upload", upload.single("pdfFile"), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: "No file uploaded." });
    }

    // Access file data in memory (req.file.buffer)
    const fileData = req.file.buffer;

    console.log("Uploaded file:", req.file);  // Log file info
    console.log("File data (buffer):", fileData);  // Log the file data (binary buffer)

    // Process the file here (for example, convert, analyze, or send elsewhere)

    res.send("File uploaded successfully without saving.");
});

// Start the server
const PORT = 4000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
