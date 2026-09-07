import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from extraction import run_extraction
from auth import verify_user
from lensai import ask_question

app = FastAPI()

document_status: dict[str, dict] = {}

origins = [
    "https://www.openlens.space",
    "https://openlens.space",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def status():
    return {"Status": "Connected"}


@app.get("/documents/{document_id}")
async def get_document(document_id: str, request: Request):
    user_id, err = verify_user(request)
    if err:
        raise HTTPException(status_code=401, detail=err["message"])

    if document_id not in document_status:
        raise HTTPException(status_code=404, detail="Document not found")
    return document_status[document_id]


@app.post("/documents/{document_id}/retry")
async def retry_document(document_id: str, request: Request):
    user_id, err = verify_user(request)
    if err:
        raise HTTPException(status_code=401, detail=err["message"])

    data = await request.json()
    image_path = data.get("image_path")
    model_name = data.get("model")

    if not image_path:
        raise HTTPException(status_code=400, detail="image_path is required")
    if not model_name:
        raise HTTPException(status_code=400, detail="model is required")

    document_status[document_id] = {"status": "processing"}
    asyncio.create_task(_run_extraction_bg(image_path, model_name, document_id, user_id))

    return {"status": "processing", "document_id": document_id}


@app.post("/imgpip")
async def process_img(request: Request):
    user_id, err = verify_user(request)
    if err:
        raise HTTPException(status_code=401, detail=err["message"])

    data = await request.json()
    document_id = data.get("document_id")
    model_name = data.get("model")
    image_path = data.get("image_path")

    if not image_path:
        raise HTTPException(status_code=400, detail="image_path is required")
    if not model_name:
        raise HTTPException(status_code=400, detail="model is required")
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id is required")

    document_status[document_id] = {"status": "processing"}
    asyncio.create_task(_run_extraction_bg(image_path, model_name, document_id, user_id))

    return {"status": "processing", "document_id": document_id}


@app.post("/ask")
async def ask(request: Request):
    user_id, err = verify_user(request)
    if err:
        raise HTTPException(status_code=401, detail=err["message"])

    data = await request.json()
    question = data.get("question")
    context = data.get("context")
    document_id = data.get("document_id")

    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if context is None:
        raise HTTPException(status_code=400, detail="context is required")

    result = ask_question(question, context, document_id)
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    return result


async def _run_extraction_bg(image_path: str, model_name: str, document_id: str, user_id: str):
    """Background task that runs extraction and updates in-memory status."""
    try:
        result = await run_extraction(image_path, model_name, document_id, user_id)
        document_status[document_id] = result
    except Exception as e:
        document_status[document_id] = {
            "status": "extraction_failed",
            "message": f"Background task crashed: {str(e)}"
        }
