import asyncio
import time
import websockets

async def get_write_only_admin_connection(SERVER_PORT, VERBOSE):
    admin_connection = None
    # wait until admin connection is up
    while True:
        time.sleep(0.5)
        try:
            admin_connection = await websockets.client.connect(f"ws://127.0.0.1:{SERVER_PORT}/?role=admin")
            break
        except Exception as e:
            if VERBOSE: print(e)
    # just ignore all incoming messages on admin connection
    async def admin_reader():
        try:
            async for message in admin_connection:
                pass
        except Exception as e:
            if VERBOSE: print(e)
    asyncio.ensure_future(admin_reader())
    return admin_connection