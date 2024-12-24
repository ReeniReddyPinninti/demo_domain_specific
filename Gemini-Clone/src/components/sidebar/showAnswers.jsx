import { useContext, useState, useEffect } from "react";
import { Context } from "../../context/Context";
import { useAuthContext } from "../../reducer/useAuthContext";
import { showSavedAns } from "../../services/Apis";
import { useNavigate } from "react-router-dom";

const ShowAnswers = () => {
    const [userData, setUserData] = useState([]);
    const [error, setError] = useState('');
    const { user } = useAuthContext();
    const navigate = useNavigate();
  
    const fetchUserData = async () => {
      try {
        const email = user?.email;
        if (email) {
          const newdata = { email: email };
          const response = await showSavedAns(newdata);
          if (Array.isArray(response.data)) {
            setUserData(response.data);
          } else {
            setUserData([]);
            setError('Unexpected response format.');
          }
          setError('');
        } else {
          setError('No email found in user context.');
        }
      } catch (err) {
        console.error("Error fetching user data:", err);
        setError(err.response?.data?.error || 'An error occurred.');
        setUserData([]);
      }
    };
  
    useEffect(() => {
      fetchUserData(); 
    }, []); 
  
    return (
        <div id="answers-container">
        <button onClick={() => navigate('/')}>Go to home</button>
        {error && (
            <div id="error-message" style={{ color: 'red' }}>
                {error}
            </div>
        )}
        <ul id="user-data-list">
            {userData.map((item, index) => (
                <li key={index}>
                    {typeof item === 'string' ? item : JSON.stringify(item.answer, null, 2)}
                </li>
            ))}
        </ul>
    </div>
    );
  };
  
  export default ShowAnswers;