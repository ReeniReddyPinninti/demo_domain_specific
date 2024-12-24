import "./sidebar.css";
import { assets } from "../../assets/assets";
import { useContext, useState, useEffect } from "react";
import { Context } from "../../context/Context";
import { useAuthContext } from "../../reducer/useAuthContext";
import { getData } from "../../services/Apis";
import { useNavigate } from "react-router-dom";

const Sidebar = () => {
    const [extended, setExtended] = useState(false);
    const { onSent, prevPrompts, setRecentPrompt, newChat, addLink } = useContext(Context);
    const [userData, setUserData] = useState([]);
    const [error, setError] = useState('');
    const { user } = useAuthContext();
    const navigate = useNavigate();

    const loadPreviousPrompt = async (prompt) => {
        setRecentPrompt(prompt);
        await onSent(prompt);
    };

    const fetchUserData = async () => {
        try {
            const email = user.email;
            if (email) {
                const newdata = { email: email };
                const response = await getData(newdata);
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
        if (extended) {
            fetchUserData();
        }
    }, [extended]);

    return (
        <div className="sidebar">
            <div className="top">
                <img
                    src={assets.menu_icon}
                    className="menu"
                    alt="menu-icon"
                    onClick={() => setExtended(prev => !prev)}
                />
                <div className="sidebar-container">
                    <div className="new-chat">
                        <img src={assets.plus_icon} alt="" onClick={() => newChat()} />
                        {extended ? <p onClick={() => navigate('/showAns')}>Show saved answers</p> : null}
                    </div>
                </div>
                {extended && (
                    <div className="recent">
                        <p className="recent-title">Recent</p>
                        <div className="user-data">
                            <h3>User Data</h3>
                            {error && <p style={{ color: 'red' }}>{error}</p>}
                            {Array.isArray(userData) && userData.length > 0 ? (
                                <div>
                                    {userData.map((dataItem, index) => (
                                        <div key={index}>
                                            <h4>Data {index + 1}</h4>
                                            <pre>{JSON.stringify(dataItem.data, null, 2)}</pre> 
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <p>No user data found.</p>
                            )}
                        </div>
                    </div>
                )}
            </div>
            <div className="bottom">
                <div className="bottom-item recent-entry">
                    <img src={assets.question_icon} alt="" />
                    {extended ? <p>Help</p> : null}
                </div>
                <div className="bottom-item recent-entry">
                    <img src={assets.history_icon} alt="" />
                    {extended ? <p>Activity</p> : null}
                </div>
                <div className="bottom-item recent-entry">
                    <img src={assets.setting_icon} alt="" />
                    {extended ? <p>Settings</p> : null}
                </div>
            </div>
        </div>
    );
};

export default Sidebar;
