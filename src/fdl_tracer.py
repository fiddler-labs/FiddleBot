from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.http.trace_exporter import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

import utils
import constants


class FdlTracer:
    _instance = None

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    def __init__(self, tracer_name: str, otel_export: str):
        """
        Initialize the FdlTracer with the given tracer name and otel export.
        """
        if otel_export == constants.OTEL_EXP_CONSOLE:
            self.provider = TracerProvider()
            self.processor = SimpleSpanProcessor(ConsoleSpanExporter())
            self.provider.add_span_processor(self.processor)

            trace.set_tracer_provider(self.provider)

        elif otel_export == constants.OTEL_EXP_COLLECTOR:
            self.resource = Resource.create(
                attributes={
                    SERVICE_NAME: constants.SERVICE_NAME,
                    constants.OTEL_RESOURCE_APP_ID: constants.APPLICATION_ID,
                }
            )
            self.provider = TracerProvider(resource=self.resource)
            trace.set_tracer_provider(self.provider)
            self.processor = SimpleSpanProcessor(
                OTLPSpanExporter(
                    endpoint=constants.OTEL_ENDPOINT,
                    headers=constants.OTEL_HEADER,
                    compression=Compression.Gzip,
                )
            )
            trace.get_tracer_provider().add_span_processor(self.processor)
            for key, value in constants.OTEL_HEADER.items():
                print(f"{key}: {value}")

        self.tracer = trace.get_tracer(tracer_name)

    def get_tracer(self):
        """Get the tracer"""
        return self.tracer
