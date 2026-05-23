import { router, publicProcedure } from '../trpc';
import { z } from 'zod';

const QualityLevel = z.enum(['Very low', 'Low', 'Average', 'High', 'Very high']);
const ThreeLevelQuality = z.enum(['Low', 'Average', 'High']);
const IssueSubtype = z.enum(['Subjective', 'Ambiguous', 'Nonverifiable', 'Negative', 'Vague']);
const AIModel = z.enum(['GPT-5', 'Claude', 'Qwen 3']);

export const requirementsRouter = router({
  // Get all requirements
  list: publicProcedure
    .query(async ({ ctx }) => {
      const requirements = await ctx.prisma.requirement.findMany({
        orderBy: { createdAt: 'desc' },
      });
      return requirements;
    }),

  // Get single requirement
  get: publicProcedure
    .input(z.object({ id: z.string() }))
    .query(async ({ ctx, input }) => {
      const requirement = await ctx.prisma.requirement.findUnique({
        where: { id: input.id },
        include: { modifications: true },
      });
      return requirement;
    }),

  // Create new requirement
  create: publicProcedure
    .input(z.object({
      text: z.string().min(1),
      quality: z.string().optional(),
    }))
    .mutation(async ({ ctx, input }) => {
      const requirement = await ctx.prisma.requirement.create({
        data: {
          text: input.text,
          quality: input.quality || 'Medium',
        },
      });
      return requirement;
    }),

  // Update requirement
  update: publicProcedure
    .input(z.object({
      id: z.string(),
      text: z.string().optional(),
      quality: z.string().optional(),
    }))
    .mutation(async ({ ctx, input }) => {
      const { id, ...data } = input;
      const requirement = await ctx.prisma.requirement.update({
        where: { id },
        data,
      });
      return requirement;
    }),

  // Delete requirement
  delete: publicProcedure
    .input(z.object({ id: z.string() }))
    .mutation(async ({ ctx, input }) => {
      await ctx.prisma.requirement.delete({
        where: { id: input.id },
      });
      return { success: true };
    }),

  // Create modification
  createModification: publicProcedure
    .input(z.object({
      requirementId: z.string(),
      stakeholderFeedback: z.string().optional(),
      modifiedText: z.string(),
      aiModel: AIModel,
      detectedSubtypes: z.array(IssueSubtype),
      qualityOfModification: QualityLevel,
      correctnessOfChange: QualityLevel,
      aiModelComment: z.string().optional(),
      reqQuality: ThreeLevelQuality,
      changeQuality: ThreeLevelQuality,
      finalResult: QualityLevel,
    }))
    .mutation(async ({ ctx, input }) => {
      const modification = await ctx.prisma.requirementModification.create({
        data: input,
      });
      return modification;
    }),

  // Get modifications for a requirement
  getModifications: publicProcedure
    .input(z.object({ requirementId: z.string() }))
    .query(async ({ ctx, input }) => {
      const modifications = await ctx.prisma.requirementModification.findMany({
        where: { requirementId: input.requirementId },
        orderBy: { createdAt: 'desc' },
      });
      return modifications;
    }),

  // Detect issues in a requirement text using AI service
  detectRequirementIssues: publicProcedure
    .input(z.object({ 
      requirementText: z.string().min(1),
    }))
    .query(async ({ ctx, input }) => {
      // Use the HuggingFaceService to detect issues with probabilities
      const issueResult = await ctx.huggingFace.detectIssuesWithProbabilities(input.requirementText);

      // Extract probabilities for the 5 defect types (default to 0 if not present)
      const allProbs = issueResult.all_probabilities;
      const subjective = allProbs['Subjective'] || 0;
      const ambiguous = allProbs['Ambiguous'] || 0;
      const nonverifiable = allProbs['Nonverifiable'] || 0;
      const negative = allProbs['Negative'] || 0;
      const vague = allProbs['Vague'] || 0;

      // Calculate defect severity using fuzzy inference system
      const defectSeverity = await ctx.huggingFace.calculateDefectSeverity(
        subjective,
        ambiguous,
        nonverifiable,
        negative,
        vague
      );

      return {
        requirementText: input.requirementText,
        detectedIssues: issueResult.predicted_labels,
        probabilities: issueResult.all_probabilities,
        defectSeverity: defectSeverity.defect_severity,
        qualityLevel: defectSeverity.defect_severity_label,
        timestamp: new Date().toISOString(),
      };
    }),

  // Find requirement defects using AI evaluation
  // Qwen evaluates the modification, SetFit detects quality issues
  findRequirementDefects: publicProcedure
    .input(z.object({ 
      requirementId: z.string(),
      original: z.string().min(1),
      modification: z.string().default(''),
      modified: z.string().min(1), // Required - the modified requirement to evaluate
    }))
    .query(async ({ ctx, input }) => {
      // Use Qwen to evaluate modification scores, SetFit to detect issues
      const analysisResult = await ctx.huggingFace.analyzeRequirement(
        input.original,
        input.modification,
        input.modified || input.original // Fallback to original if not provided
      );

      console.log({ analysisResult })

      // Map labels from backend (very_low, low, average, high, very_high) to frontend format (Very low, Low, Average, High, Very high)
      const mapQualityLabel = (label: string | undefined): z.infer<typeof QualityLevel> => {
        if (!label) return 'Average';
        const normalized = label.toLowerCase();
        if (normalized === 'very_low' || normalized === 'very low') return 'Very low';
        if (normalized === 'low') return 'Low';
        if (normalized === 'average') return 'Average';
        if (normalized === 'high') return 'High';
        if (normalized === 'very_high' || normalized === 'very high') return 'Very high';
        return 'Average';
      };

      // Map 3-level correctness label (low/average/high) to frontend format
      const mapThreeLevelLabel = (label: string | undefined): 'Low' | 'Average' | 'High' => {
        console.log('Mapping 3-level label:', label);
        if (!label) {
          console.log('Label is undefined/null, returning Average');
          return 'Average';
        }
        const normalized = label.toLowerCase();
        console.log('Normalized label:', normalized);
        if (normalized === 'low') return 'Low';
        if (normalized === 'average') return 'Average';
        if (normalized === 'high') return 'High';
        console.log('Label not recognized, returning Average');
        return 'Average';
      };

      const correctnessThreeLevelMapped = mapThreeLevelLabel(analysisResult.parsedAnalysis?.correctness_three_level);
      console.log('correctness_three_level from parsedAnalysis:', analysisResult.parsedAnalysis?.correctness_three_level);
      console.log('correctnessThreeLevel mapped:', correctnessThreeLevelMapped);

      return {
        requirementId: input.requirementId,
        original: input.original,
        modification: input.modification,
        modified: input.modified || input.original, // Return the provided modified requirement
        rawAnalysis: analysisResult.rawAnalysis,
        preservationCorrectness: analysisResult.parsedAnalysis?.preservation_correctness ?? null,
        changeCorrectness: analysisResult.parsedAnalysis?.change_correctness ?? null,
        qualityOfChange: analysisResult.parsedAnalysis?.quality_of_change ?? null,
        qualityOfChangeLabel: mapQualityLabel(analysisResult.parsedAnalysis?.quality_of_change_label),
        correctnessThreeLevel: correctnessThreeLevelMapped,
        modifiedQualityLevel: analysisResult.parsedAnalysis?.modified_quality_level ?? null,
        finalResult: mapQualityLabel(analysisResult.parsedAnalysis?.final_result_label),
        finalResultScore: analysisResult.parsedAnalysis?.final_result ?? null,
        detectedIssues: analysisResult.parsedAnalysis?.detected_issues ?? [],
        comments: analysisResult.parsedAnalysis?.comments ?? [],
        timestamp: new Date().toISOString(),
      };
    }),
});

