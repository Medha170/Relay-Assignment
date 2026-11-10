
import asyncio
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()


async def main():
    async with AsyncGroq() as client:
        models = await client.models.list()

        for model in models.data:
            print(model.id)


asyncio.run(main())
