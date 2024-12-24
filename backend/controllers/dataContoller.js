// const userdata = require("../schema/userData");
// const userAns = require("../schema/saveans");
// const mongoose = require("mongoose");
// const axios = require('axios');

// exports.UserData = async (req,res) => {
//     const {email, data} = req.body;
    
//     try {
//         const response = await axios.post('http://0.0.0.0:5100/embeddings', {email, data});
//         const answer = response.data.message; 
//         console.log(answer);
//         const savedata = userdata({
//             email, data
//         });
//         const storedate = await savedata.save();
//         //res.status(200).json({storedate});
//         res.json({answer});
//     }
//     catch (error){
//         res.status(400).json({error : "Error saving the link"});
//     }
// };

// exports.getData = async (req,res) => {
//     const {email} =  req.body;

//     try {
//         const users = await userdata.find({email});
//         if (users.length===0)
//             {
//                 return res.status(400).send({ error: 'No users found with this email.' });
//             }
//         res.send(users);
//     } catch (error) {
//         res.status(400).send({ error: 'Server error.' });
//     }
// };

// exports.query = async (req, res) => {
//     try {
//         const { username, query } = req.body;
//         if (!username || !query) {
//             return res.status(400).json({ error: "Both name and query must be provided" });
//         }
        
//         // Process your logic here
        
//         //const answer = "hello";
//         const response = await axios.post('http://0.0.0.0:5100/getResponse', {username, query});
//         const answer = response.data.message; 
//         console.log(answer);
//         //res.end( JSON.stringify(answer.message) );
//         res.json({answer});
//     } catch (error) {
//         console.error("Error in query handler:", error);
//         res.status(500).json({ error: "Internal server error" });
//     }
// };

// exports.saveAns = async(req,res) =>{
//     try {
//         const { email, answer } = req.body;
//         const newAnswer = new userAns({ email, answer });
//         await newAnswer.save();
//         res.status(200).json({ message: 'Answer saved successfully' });
//     } catch (error) {
//         console.error('Error saving answer:', error);
//         res.status(500).json({ message: 'Error saving answer' });
//     }
// };

// exports.getAns = async(req,res) =>{
//     const {email} =  req.body;

//     try {
//         const users = await userAns.find({email});
//         if (users.length===0)
//             {
//                 return res.status(400).send({ error: 'No users found with this email.' });
//             }
//         res.send(users);
//     } catch (error) {
//         res.status(400).send({ error: 'Server error.' });
//     }
// };

// const userdata = require("../schema/userData");

// const multer = require('multer');
// const axios = require('axios');

// // Set up multer to store the file in memory
// const storage = multer.memoryStorage();
// const upload = multer({ storage });

// // Define the route for processing the file
// exports.UserData = [
//     upload.single('file'),  // Handle single file upload
//     async (req, res) => {
//         const { email } = req.body;  // Get the email from the body
//         const file = req.file;  // Get the file from memory

//         if (!file) {
//             return res.status(400).json({ error: "No file uploaded" });
//         }

//         try {
//             // Send the file as binary data to FastAPI for processing
//             const response = await axios.post('http://0.0.0.0:5100/embeddings', {
//                 email,
//                 file: file.buffer,  // Send file buffer to FastAPI
//             }, {
//                 headers: {
//                     'Content-Type': 'application/json',  // Ensure proper content type
//                 }
//             });

//             const answer = response.data.message;  // Response from FastAPI

//             // Optionally save other details to MongoDB (excluding file)
//             const savedata = new userdata({ email });
//             await savedata.save();

//             // Send response back to the client
//             res.json({ answer });
//         } catch (error) {
//             console.error('Error processing file:', error);
//             res.status(400).json({ error: "Error processing the file" });
//         }
//     }
// ];

// exports.query = async (req, res) => {
//     const { username, query } = req.body;
//     if (!username || !query) {
//         return res.status(400).json({ error: "Both name and query must be provided" });
//     }

//     try {
//         // Query FastAPI for a response
//         const response = await axios.post('http://0.0.0.0:5100/getResponse', { username, query });
//         const answer = response.data.response; // Response from FastAPI
        
//         res.json({ answer });
//     } catch (error) {
//         console.error("Error in query handler:", error);
//         res.status(500).json({ error: "Internal server error" });
//     }
// };

// dataController.js
const multer = require('multer');
const axios = require('axios');

// Set up multer to store the file in memory
const storage = multer.memoryStorage();
const upload = multer({ storage });

// Export UserData as an array of middleware and handler
exports.UserData = [
    upload.single('file'),  // Handle single file upload
    async (req, res) => {
        const { email } = req.body;  // Get the email from the body
        const file = req.file;  // Get the file from memory

        if (!file) {
            return res.status(400).json({ error: "No file uploaded" });
        }

        try {
            // Send the file as binary data to FastAPI for processing
            const response = await axios.post('http://0.0.0.0:5100/embeddings', {
                username: email,  // Send the username (email) as part of the payload
                pdf_file: file.buffer,  // Send file buffer to FastAPI
            }, {
                headers: {
                    'Content-Type': 'application/json',  // Ensure proper content type
                }
            });

            const message = response.data.message;  // Response from FastAPI

            // Send response back to the client
            res.json({ message });
        } catch (error) {
            console.error('Error processing file:', error);
            res.status(400).json({ error: "Error processing the file" });
        }
    }
];

// Query handler (not changed)
exports.query = async (req, res) => {
    const { username, query } = req.body;
    if (!username || !query) {
        return res.status(400).json({ error: "Both name and query must be provided" });
    }

    try {
        // Query FastAPI for a response
        const response = await axios.post('http://0.0.0.0:5100/getResponse', { username, query });
        const answer = response.data.response; // Response from FastAPI
        
        res.json({ answer });
    } catch (error) {
        console.error("Error in query handler:", error);
        res.status(500).json({ error: "Internal server error" });
    }
};
