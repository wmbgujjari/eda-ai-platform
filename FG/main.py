from fastapi import FastAPI
from FG.routers import consumption_routes, daily_demand_routes,schedule_routes,dynamic_routes, segmentation_routes,demand_routes,revenue_routes
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more verbosity
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),                     # Console log
        logging.FileHandler("FG/core/logs/app.log")          # File log
    ]
)


app = FastAPI(title="Power Sector Forecasting API")

# Add CORS middleware right after app initialization
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8280"],  # Use ["http://localhost:5500"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include namespaced routes
app.include_router(consumption_routes.router)
app.include_router(schedule_routes.router)
app.include_router(dynamic_routes.router)
app.include_router(segmentation_routes.router)
app.include_router(demand_routes.router)
app.include_router(revenue_routes.router)
app.include_router(daily_demand_routes.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


