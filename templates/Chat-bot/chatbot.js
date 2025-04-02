const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");
const chatbox = document.querySelector(".chatbox");
const chatbotToggler = document.querySelector(".chatbot-toggler")

let userMessage;

const createChatLi = (message , className)=>{
    //สร้างแชท <li> element ผ่าน message และ className
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat",className);
    let chatContent = className === "outgoing" ? `<p></p>`: `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatLi.innerHTML = chatContent;
    chatLi.querySelector('p').textContent = message;
    return chatLi;
}

const generateResponse = (incomingChatLi) =>{
    const API_URL ="https://openrouter.ai/api/v1/chat/completions";
    const messageElement = incomingChatLi.querySelector("p");

 
    const requestoptions = {
        method: 'POST',
        headers: {
            Authorization: 'Bearer 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',
            'HTTP-Referer': 'https://www.sitename.com',
            'X-Title': 'SiteName',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: 'deepseek/deepseek-r1:free',
            messages: [{ role: 'user', content: userMessage }]
        })
    }

    fetch(API_URL , requestoptions).then(res => res.json()).then(data => {
        messageElement.textContent = data.choices[0].message.content;
    }).catch(()=>{
        messageElement.textContent = "Oops! Something wrong. Pleas try again.";
    }).finally(() => chatbox.scrollTo(0,chatbox.scrollHeight));
}

const handleChat = ()=>{
    userMessage = chatInput.value.trim();
    if(!userMessage) return;
    chatInput.value = ""

    //เพิ่มข้อความของ user ไปที่ chatbox
    chatbox.appendChild(createChatLi(userMessage , 'outgoing'));
    chatbox.scrollTo(0,chatbox.scrollHeight);

    setTimeout(()=>{
        //แสดงข้อความ Thinking... ขณะรอ bot response
        const incomingChatLi = createChatLi("Thinking..." , 'incoming');
        chatbox.appendChild(incomingChatLi);
        chatbox.scrollTo(0,chatbox.scrollHeight);
        generateResponse(incomingChatLi);
    },600)
}


sendChatBtn.addEventListener("click",handleChat);
// chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));
chatbotToggler.addEventListener("click", () => {
    document.querySelector("#show-chatbot").classList.toggle("show-chatbot");
});