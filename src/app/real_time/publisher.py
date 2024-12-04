""" Kafka publisher. """
import asyncio
from aiokafka import AIOKafkaProducer
from src.app.real_time.message import RealTimeKPI


class KafkaPublisher:

    instance = None

    def __init__(self, topic, port, servers) -> None:
        self._topic = topic
        self._port = port
        self._servers = servers
        self.aioproducer = self.create_kafka()

    def create_kafka(self):
        loop = asyncio.get_event_loop()
        return AIOKafkaProducer(
            loop=loop,
            bootstrap_servers=f'{self._servers}:{self._port}'
        )

    async def open_session(self):
        self.aioproducer.start()
        return 'Kafka producer started'

    async def send(self, data: RealTimeKPI):
        try:
            topic_name = self._topic
            await self.aioproducer.send_and_wait(topic_name, data.to_json())
        except Exception as e:
            await self.aioproducer.stop()
            raise e
        return 'Message sent successfully'

    async def finalize(self):
        await self.aioproducer.stop()
        return 'Kafka producer stopped'
