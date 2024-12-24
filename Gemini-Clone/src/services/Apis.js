import { commonrequest } from "./ApiCall";
import {BACKEND_URL} from "./helper";


export const registerfunction = async(data)=>{
    return await commonrequest("POST",`${BACKEND_URL}/signup`,data)
}

export const sentOtpFunction = async(data)=>{
    return await commonrequest("POST",`${BACKEND_URL}/otp`,data)
}

export const saveUserdata = async (data) => {
  return await commonrequest("POST", `${BACKEND_URL}/savedata`, data)
}

export const getData = async(data) =>{
  return await commonrequest("POST", `${BACKEND_URL}/sendData`, data)
}

export const getAnswer = async(data) => {
  return await commonrequest("POST", `${BACKEND_URL}/getRes`, data)
}

export const saveAnswer = async(data) => {
  return await commonrequest("POST", `${BACKEND_URL}/saveAnswer`, data)
}

export const showSavedAns = async(data) => {
  return await commonrequest("POST", `${BACKEND_URL}/getAnswers`, data)
}


export const userVerify = (data) => {
    return fetch(`${BACKEND_URL}/signin`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
  };

  export const uploadPdf = async (file) => {
    const formData = new FormData();
    formData.append('pdfFile', file);

    const response = await fetch(`${BACKEND_URL}/extract-text`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error('Network response was not ok');
    }

    return response.text();
};

// export const addLinkFunction = async(data) => {
//   return await commonrequest("Post", `${BACKEND_URL}/addLink`, data);
// }
