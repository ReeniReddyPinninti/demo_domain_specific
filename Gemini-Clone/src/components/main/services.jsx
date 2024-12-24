import axios from 'axios';

export const uploadPdf = async (file, username) => {
    const formData = new FormData();
    formData.append("pdf", file);
    //formData.append("username", username); // Send username as well
    const response = await axios.post("http://127.0.0.1:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
    });
    return response;
};

export const getAnswer = async (data) => {
    const response = await axios.post("http://127.0.0.1:5000/generate", data);
    return response;
};
