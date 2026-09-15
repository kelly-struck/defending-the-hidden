import os
from typing import Any, List, Dict, Union
from openai import AsyncOpenAI
from agentscope.model import ChatModelBase, ChatResponse
from agentscope.message import Msg, TextBlock

class IdentityFormatter:
    async def format(self, msgs: List[Msg]) -> List[Msg]:
        return msgs

class DoubaoModel(ChatModelBase):
    def __init__(self, config_name: str = "doubao-1.5-vision-pro-250328", api_key: str = None, **kwargs):
        super().__init__(model_name=config_name, stream=False)
        self.client = AsyncOpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=api_key or os.environ.get("ARK_API_KEY"),
        )

    async def __call__(self, messages: List[Dict] | List[Msg], **kwargs) -> ChatResponse:
        # Convert AgentScope messages to OpenAI format
        openai_messages = []
        
        # Ensure messages is a list
        if not isinstance(messages, list):
            messages = [messages]

        for msg in messages:
            # Handle Msg object or dict
            role = getattr(msg, "role", None) or msg.get("role", "user")
            content_raw = getattr(msg, "content", None) or msg.get("content", "")
            
            openai_content = []
            
            if isinstance(content_raw, str):
                openai_content.append({"type": "text", "text": content_raw})
            elif isinstance(content_raw, list):
                for block in content_raw:
                    # Handle TextBlock/ImageBlock objects or dicts
                    b_type = getattr(block, "type", None) or block.get("type")
                    
                    if b_type == "text":
                        text_val = getattr(block, "text", None) or block.get("text", "")
                        openai_content.append({"type": "text", "text": text_val})
                    elif b_type == "image":
                        # AgentScope ImageBlock source might be a dict or object
                        source = getattr(block, "source", None) or block.get("source")
                        url = None
                        
                        if isinstance(source, dict):
                            if "url" in source:
                                url = source["url"]
                            elif "data" in source and "media_type" in source:
                                # Handle base64 data
                                # Format: data:image/png;base64,......
                                url = f"data:{source['media_type']};base64,{source['data']}"
                        elif hasattr(source, "url"):
                            url = source.url
                        else:
                            # Fallback: assume source is the url string itself if it's a string
                            if isinstance(source, str):
                                url = source
                        
                        if url:
                            openai_content.append({
                                "type": "image_url",
                                "image_url": {"url": url}
                            })
            
            openai_messages.append({
                "role": role,
                "content": openai_content
            })

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                **kwargs
            )
            
            # Extract text response
            response_text = response.choices[0].message.content
            
            # Return ChatResponse (AgentScope expects this)
            return ChatResponse(content=[TextBlock(type="text", text=response_text)])
            
        except Exception as e:
            # Return an error message in the response content so it's logged
            return ChatResponse(content=[TextBlock(type="text", text=f"Error calling Doubao: {str(e)}")])

