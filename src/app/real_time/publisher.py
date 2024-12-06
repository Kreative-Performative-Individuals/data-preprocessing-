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
        KafkaPublisher.instance = self

    def create_kafka(self):
        loop = asyncio.get_event_loop()

        return AIOKafkaProducer(
            loop=loop,
            bootstrap_servers=f'{self._servers}:{self._port}'
        )

    async def send(self, data: list[RealTimeKPI], stop_event):
        try:
            await self.aioproducer.send_and_wait(self._topic, data)
        except Exception as e:
            await self.aioproducer.stop()
            stop_event.set()
        return 'Message sent successfully'

    async def finalize(self):
        await self.aioproducer.stop()
        return 'Kafka producer stopped'

    @property
    def topic(self):
        return self._topic
