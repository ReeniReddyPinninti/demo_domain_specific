import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Main from './components/main/Main';
import Sidebar from './components/sidebar/Sidebar';
import SignIn from './components/login/signIn';
import SignUp from './components/login/signUp';
import Otp from './components/login/otp';
import { AuthContextProvider } from './reducer/useReducer';
import { useAuthContext } from './reducer/useAuthContext';
import { Navigate } from 'react-router-dom';
import 'react-toastify/dist/ReactToastify.css';
import ShowAnswers from './components/sidebar/showAnswers';

const Routing = () => {
  const { user } = useAuthContext();

  return (
   
    <Router>
      {user && <Sidebar/>}
      <Routes>
        {/* <Route path="/" element={<Main />} /> */}
        {/* <Route path="/signIn" element={<SignIn />} /> */}
        <Route path="/" element={user ? <Main /> : <Navigate to="/signIn" />}/>
        <Route path="/signIn" element={!user ? <SignIn /> : <Navigate to="/" />}/>
        <Route path="/signUp" element={<SignUp />} />
        {/* <Route path="/otp" element={<Otp/>} /> */}
        <Route path="otp" element={!user ? <Otp />: <Navigate to="/" />}/>
        {/* <Route path="/extract-text" element={<Main />} /> */}
        <Route path = "showAns" element={<ShowAnswers/>}/>
      </Routes>
    </Router>
  );
};

const App = () =>{
  return (
    <AuthContextProvider>
      <Routing/>
    </AuthContextProvider>
  );
};

export default App;
