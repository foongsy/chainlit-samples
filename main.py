from dataclasses import dataclass
import os
from typing import Any
import chainlit as cl
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from httpx import AsyncClient

load_dotenv()

@dataclass
class Deps:
    client: AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None


model = OpenAIModel(
    'google/gemini-2.0-flash-lite-001',
    provider=OpenAIProvider(
        base_url='https://openrouter.ai/api/v1',
        api_key=os.getenv("OPENROUTER_API_KEY"),
        http_client=AsyncClient(verify=False)
    ),
)

# Instrument the OpenAI client
# cl.instrument_openai()

manager_prompt = (
    'You are a manager of a team of agents.'
    'You are responsible for delegating tasks to the agents.'
    'Use `weather_factory` to answer weather related question when a location is given.'
    'Answer directly if the topic is not about weather.'
    'Answer everything in Hong Kong Traditional Chinese only.'
)
""" Manager agent """
manager_agent = Agent(
    model=model,
    system_prompt=manager_prompt,
    deps_type=Deps,
)

@manager_agent.tool
async def weather_factory(ctx: RunContext[Deps], location: str) -> str:
    response = await weather_agent.run(location, deps=ctx.deps)
    return response.output

""" Weather Agent """
weather_agent = Agent(
    model=model,
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    system_prompt=(
        'Be concise, reply with one sentence.'
        'Use the `get_lat_lng` tool to get the latitude and longitude of the locations, '
        'ensure the location is given in English as the `get_lat_lng` tool only understand English,'
        'then use the `get_weather` tool to get the weather.'
    ),
    deps_type=Deps,
    retries=2,
    instrument=True,
)


@weather_agent.tool
async def get_lat_lng(
    ctx: RunContext[Deps], location_description: str
) -> dict[str, float]:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        location_description: A description of a location.
    """
    if ctx.deps.geo_api_key is None:
        # if no API key is provided, return a dummy response (London)
        return {'lat': 51.1, 'lng': -0.1}

    params = {
        'q': location_description,
        'api_key': ctx.deps.geo_api_key,
    }
    #with logfire.span('calling geocode API', params=params) as span:
    r = await ctx.deps.client.get('https://geocode.maps.co/search', params=params)
    r.raise_for_status()
    data = r.json()
    # span.set_attribute('response', data)

    if data:
        return {'lat': data[0]['lat'], 'lng': data[0]['lon']}
    else:
        raise ModelRetry('Could not find the location')


@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location.

    Args:
        ctx: The context.
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    if ctx.deps.weather_api_key is None:
        # if no API key is provided, return a dummy response
        return {'temperature': '21 °C', 'description': 'Sunny'}

    params = {
        'apikey': ctx.deps.weather_api_key,
        'location': f'{lat},{lng}',
        'units': 'metric',
    }
    #with logfire.span('calling weather API', params=params) as span:
    r = await ctx.deps.client.get(
        'https://api.tomorrow.io/v4/weather/realtime', params=params
    )
    r.raise_for_status()
    data = r.json()
    # span.set_attribute('response', data)

    values = data['data']['values']
    # https://docs.tomorrow.io/reference/data-layers-weather-codes
    code_lookup = {
        1000: 'Clear, Sunny',
        1100: 'Mostly Clear',
        1101: 'Partly Cloudy',
        1102: 'Mostly Cloudy',
        1001: 'Cloudy',
        2000: 'Fog',
        2100: 'Light Fog',
        4000: 'Drizzle',
        4001: 'Rain',
        4200: 'Light Rain',
        4201: 'Heavy Rain',
        5000: 'Snow',
        5001: 'Flurries',
        5100: 'Light Snow',
        5101: 'Heavy Snow',
        6000: 'Freezing Drizzle',
        6001: 'Freezing Rain',
        6200: 'Light Freezing Rain',
        6201: 'Heavy Freezing Rain',
        7000: 'Ice Pellets',
        7101: 'Heavy Ice Pellets',
        7102: 'Light Ice Pellets',
        8000: 'Thunderstorm',
    }
    return {
        'temperature': f'{values["temperatureApparent"]:0.0f}°C',
        'description': code_lookup.get(values['weatherCode'], 'Unknown'),
    }
""" Weather agent end """

@cl.on_chat_start
def on_start():
    deps = Deps(
        client=AsyncClient(verify=False),
        weather_api_key=os.getenv('WEATHER_API_KEY'),
        geo_api_key=os.getenv('GEO_API_KEY')
    )
    cl.user_session.set("agent", manager_agent)
    cl.user_session.set("deps", deps)

@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    deps = cl.user_session.get("deps")
    response = agent.run_sync(message.content, deps=deps)
    await cl.Message(content=response.output).send()
