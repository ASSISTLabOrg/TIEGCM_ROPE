"""
Main file.

Contact: Violet Player
Email: violet.player@colorado.edu
"""
import asyncio
from concurrent.futures import ProcessPoolExecutor as Pool
from json import load
from models.forecast import build_forecaster

async def handle_client(reader, writer, executor):
    
    addr = writer.get_extra_info('peername')
    print(f"Connection from {addr!r}")
        
    while True:

        data = await reader.read(100)
        if not data:
            break
        
        # Offload CPU-bound task to a separate process
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            process_data,
            data
        )
        
        writer.write(result)
        await writer.drain()
    
    print(f"Close the connection from {addr!r}")
    writer.close()

def process_data(data):
    return b"Processed: " + data

async def main(settings_file):

    #### set up forecaster 
    forecaster = build_forecaster(
        load(
            settings_file
        )
    )
    
    #### initialize multiprocessing pool
    pool = Pool(
        mp_context="forkserver" # forks retain parent process variables
    )

    #### open server; anonymous client function to add args
    server = await asyncio.start_server(
        lambda x, y : handle_client(x, y, pool), 
        '127.0.0.1', 
        8888
    )

    async with server:
        print("Server started. Waiting for connections...")
        await server.serve_forever()

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    #-db DATABASE -u USERNAME -p PASSWORD -size 20
    parser.add_argument("-s", "--settings", help="Settings file")
    #parser.add_argument("-db", "--hostname", help="Database name")
    #parser.add_argument("-u", "--username", help="User name")
    #parser.add_argument("-p", "--password", help="Password")
    #parser.add_argument("-size", "--size", help="Size", type=int)
    args = parser.parse_args()
    
    asyncio.run(main(
        args.settings
        )
    )