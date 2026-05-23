import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { TrpcController } from './trpc.controller';
import { TrpcService } from './trpc.service';
import { HuggingFaceService } from '../services/huggingface.service';

@Module({
  imports: [HttpModule],
  controllers: [TrpcController],
  providers: [TrpcService, HuggingFaceService],
  exports: [TrpcService],
})
export class TrpcModule {}

