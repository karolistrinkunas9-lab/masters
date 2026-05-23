import { All, Controller, Req, Res } from '@nestjs/common';
import { TrpcService } from './trpc.service';
import { fetchRequestHandler } from '@trpc/server/adapters/fetch';

@Controller('trpc')
export class TrpcController {
  constructor(private readonly trpcService: TrpcService) {}

  @All('*')
  async handle(@Req() req: any, @Res() res: any) {
    const createContext = () => this.trpcService.getContext();
    
    // Convert Express request to Fetch API request
    const url = new URL(req.url, `http://${req.headers.host}`);
    
    const fetchRequest = new Request(url.toString(), {
      method: req.method,
      headers: req.headers,
      body:
        req.method !== 'GET' && req.method !== 'HEAD'
          ? JSON.stringify(req.body)
          : undefined,
    });

    const fetchResponse = await fetchRequestHandler({
      endpoint: '/trpc',
      req: fetchRequest,
      router: this.trpcService.getRouter(),
      createContext,
    });

    // Set response headers
    fetchResponse.headers.forEach((value, key) => {
      res.setHeader(key, value);
    });

    // Set status and send body
    res.status(fetchResponse.status);
    const body = await fetchResponse.text();
    res.send(body);
  }
}

