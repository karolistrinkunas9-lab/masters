// src/services/huggingface.service.ts
import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';

export interface RequirementAnalysis {
  preservation_correctness: number;
  change_correctness: number;
  quality_of_change?: number;
  quality_of_change_label?: string;
  correctness_three_level?: string; // Low/Average/High from FIS
  detected_issues?: string[];
  modified_quality_level?: string;
  final_result?: number;
  final_result_label?: string;
  comments: string[];
}

export interface AnalysisResult {
  rawAnalysis: string;
  parsedAnalysis: RequirementAnalysis | null;
}

export interface IssueDetectionResult {
  predicted_labels: string[];
  probabilities: Record<string, number>;
  all_probabilities: Record<string, number>;
}

export interface DefectSeverityResult {
  defect_severity: number;
  defect_severity_label: string;
}

export interface CorrectnessResult {
  correctness: number;
  correctness_label: string;
  correctness_three_level_label: string; // Low/Average/High
}

export interface RequirementQualityResult {
  requirement_quality: number;
  requirement_quality_label: string;
}

export interface RequirementAnalysisAIResult {
  preservation_correctness: number;
  change_correctness: number;
  analysis_text: string;
}


@Injectable()
export class HuggingFaceService {
  private readonly AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8001';

  constructor(private readonly httpService: HttpService) {}

  /**
   * Detect quality issues in a requirement using the Python AI service
   * Uses SetFit multi-label classification model
   */
  async detectIssues(requirementText: string): Promise<string[]> {
    const result = await this.detectIssuesWithProbabilities(requirementText);
    return result.predicted_labels;
  }

  /**
   * Detect quality issues with probabilities
   */
  async detectIssuesWithProbabilities(requirementText: string): Promise<IssueDetectionResult> {
    try {
      const { data } = await firstValueFrom(
        this.httpService.post(
          `${this.AI_SERVICE_URL}/api/inference/requirement-quality`,
          {
            texts: [requirementText]
          },
          {
            headers: { 'Content-Type': 'application/json' }
          }
        )
      );

      if (data.success && data.predictions && data.predictions.length > 0) {
        const prediction = data.predictions[0];
        return {
          predicted_labels: prediction.predicted_labels || [],
          probabilities: prediction.probabilities || {},
          all_probabilities: prediction.all_probabilities || {}
        };
      }

      return {
        predicted_labels: [],
        probabilities: {},
        all_probabilities: {}
      };
    } catch (error: any) {
      console.error('Error detecting issues from AI service:', error.message);
      return {
        predicted_labels: [],
        probabilities: {},
        all_probabilities: {}
      };
    }
  }

  /**
   * Calculate defect severity using fuzzy inference system
   */
  async calculateDefectSeverity(
    subjective: number,
    ambiguous: number,
    nonverifiable: number,
    negative: number,
    vague: number
  ): Promise<DefectSeverityResult> {
    try {
      const { data } = await firstValueFrom(
        this.httpService.post(
          `${this.AI_SERVICE_URL}/api/fuzzy/defect-severity`,
          {
            subjective,
            ambiguous,
            nonverifiable,
            negative,
            vague
          },
          {
            headers: { 'Content-Type': 'application/json' }
          }
        )
      );

      if (data.success) {
        return {
          defect_severity: data.defect_severity,
          defect_severity_label: data.defect_severity_label
        };
      }

      return {
        defect_severity: 0.5,
        defect_severity_label: 'average'
      };
    } catch (error: any) {
      console.error('Error calculating defect severity:', error.message);
      return {
        defect_severity: 0.5,
        defect_severity_label: 'average'
      };
    }
  }

  /**
   * Calculate correctness using fuzzy inference system
   */
  async calculateCorrectness(
    preservation_correctness: number,
    change_correctness: number
  ): Promise<CorrectnessResult> {
    try {
      const { data } = await firstValueFrom(
        this.httpService.post(
          `${this.AI_SERVICE_URL}/api/fuzzy/correctness`,
          {
            preservation_correctness,
            change_correctness
          },
          {
            headers: { 'Content-Type': 'application/json' }
          }
        )
      );

      if (data.success) {
        return {
          correctness: data.correctness,
          correctness_label: data.correctness_label,
          correctness_three_level_label: data.correctness_three_level_label || 'average'
        };
      }

      return {
        correctness: 0.5,
        correctness_label: 'average',
        correctness_three_level_label: 'average'
      };
    } catch (error: any) {
      console.error('Error calculating correctness:', error.message);
      return {
        correctness: 0.5,
        correctness_label: 'average',
        correctness_three_level_label: 'average'
      };
    }
  }

  /**
   * Calculate requirement quality using fuzzy inference system
   * Takes defect_severity and correctness to produce overall requirement quality
   */
  async calculateRequirementQuality(
    defect_severity: number,
    correctness: number
  ): Promise<RequirementQualityResult> {
    try {
      const { data } = await firstValueFrom(
        this.httpService.post(
          `${this.AI_SERVICE_URL}/api/fuzzy/requirement-quality`,
          {
            defect_severity,
            correctness
          },
          {
            headers: { 'Content-Type': 'application/json' }
          }
        )
      );

      if (data.success) {
        return {
          requirement_quality: data.requirement_quality,
          requirement_quality_label: data.requirement_quality_label
        };
      }

      return {
        requirement_quality: 0.5,
        requirement_quality_label: 'average'
      };
    } catch (error: any) {
      console.error('Error calculating requirement quality:', error.message);
      return {
        requirement_quality: 0.5,
        requirement_quality_label: 'average'
      };
    }
  }

  /**
   * Evaluate requirement modification using Qwen model through AI service
   * Qwen evaluates the modification and provides scores/comments only
   * Qwen does NOT generate the modified requirement
   */
  async analyzeRequirementWithQwen(
    original: string,
    modification: string,
    modified: string
  ): Promise<RequirementAnalysisAIResult> {
    const { data } = await firstValueFrom(
      this.httpService.post(
        `${this.AI_SERVICE_URL}/api/inference/requirement-analysis`,
        {
          original_requirement: original,
          modification_instruction: modification,
          modified_requirement: modified,  // Provide for Qwen to evaluate
          model_name: 'Qwen/Qwen3-4B-Instruct-2507'
        },
        {
          headers: { 'Content-Type': 'application/json' }
        }
      )
    );

    if (!data.success) {
      throw new Error(`Qwen evaluation failed: ${data.error || 'Unknown error'}`);
    }

    // Scores and comments from Qwen evaluation
    return {
      preservation_correctness: data.preservation_correctness,
      change_correctness: data.change_correctness,
      analysis_text: data.analysis_text
    };
  }

  async analyzeRequirement(
    original: string,
    modification: string,
    modified: string
  ): Promise<AnalysisResult> {
    try {
      // First, detect issues in the original requirement to get defect_severity
      const originalIssuesWithProbs = await this.detectIssuesWithProbabilities(original);
      const originalDefectSeverity = await this.calculateDefectSeverity(
        originalIssuesWithProbs.probabilities.Subjective || 0,
        originalIssuesWithProbs.probabilities.Ambiguous || 0,
        originalIssuesWithProbs.probabilities['Non-verifiable'] || 0,
        originalIssuesWithProbs.probabilities.Negative || 0,
        originalIssuesWithProbs.probabilities.Vague || 0
      );
      console.log('Original requirement defect severity:', originalDefectSeverity);
      // Use Qwen model to evaluate the modification (not generate it)
      const qwenAnalysis = await this.analyzeRequirementWithQwen(original, modification, modified);
      
      // Detect issues in the modified requirement using SetFit model
      const issuesWithProbs = await this.detectIssuesWithProbabilities(modified);
      const detectedIssues = issuesWithProbs.predicted_labels;
      console.log('Detected issues from SetFit model:', detectedIssues);
      console.log('Issue probabilities:', issuesWithProbs.probabilities);
      console.log('Qwen evaluation scores:', qwenAnalysis);

      // Calculate defect severity (quality level) for the modified requirement
      const modifiedDefectSeverity = await this.calculateDefectSeverity(
        issuesWithProbs.probabilities.Subjective || 0,
        issuesWithProbs.probabilities.Ambiguous || 0,
        issuesWithProbs.probabilities['Non-verifiable'] || 0,
        issuesWithProbs.probabilities.Negative || 0,
        issuesWithProbs.probabilities.Vague || 0
      );
      console.log('Modified requirement quality:', modifiedDefectSeverity);

      // Create analysis summary
      const rawAnalysis = qwenAnalysis.analysis_text;

      // Calculate quality_of_change using FIS (correctness) from Python service
      const correctnessResult = await this.calculateCorrectness(
        qwenAnalysis.preservation_correctness,
        qwenAnalysis.change_correctness
      );
      console.log('Quality of Change (correctness) from FIS:', correctnessResult);
      console.log('Correctness 3-level label:', correctnessResult.correctness_three_level_label);

      // Calculate quality_of_change using requirement-quality FIS
      // Takes modified requirement defect_severity and correctness
      // This combines the modified requirement's quality with correctness to get overall quality of change
      const qualityOfChangeResult = await this.calculateRequirementQuality(
        modifiedDefectSeverity.defect_severity,
        correctnessResult.correctness
      );
      console.log('Quality of Change (from modified requirement + correctness):', qualityOfChangeResult);

      // Calculate final_result using requirement-quality FIS
      // Takes original requirement defect_severity and correctness (quality_of_change)
      const requirementQualityResult = await this.calculateRequirementQuality(
        originalDefectSeverity.defect_severity,
        correctnessResult.correctness
      );
      console.log('Final Result (requirement quality) from FIS:', requirementQualityResult);

      // Return evaluation with Qwen-calculated scores, FIS-calculated quality_of_change (with 5-level label), final_result, and SetFit-detected issues
      const parsedAnalysis = {
        preservation_correctness: qwenAnalysis.preservation_correctness,
        change_correctness: qwenAnalysis.change_correctness,
        quality_of_change: correctnessResult.correctness,
        quality_of_change_label: qualityOfChangeResult.requirement_quality_label, // Combined result from modified requirement + correctness
        correctness_three_level: correctnessResult.correctness_three_level_label,
        detected_issues: detectedIssues,
        modified_quality_level: modifiedDefectSeverity.defect_severity_label,
        final_result: requirementQualityResult.requirement_quality,
        final_result_label: requirementQualityResult.requirement_quality_label,
        comments: detectedIssues.length > 0 
          ? [`Detected ${detectedIssues.length} quality issue(s) in the modified requirement`]
          : ['No quality issues detected in the modified requirement']
      };
      console.log('Parsed analysis with correctness_three_level:', parsedAnalysis.correctness_three_level);
      return {
        rawAnalysis,
        parsedAnalysis,
      };
    } catch (error: any) {
      console.error('Error evaluating requirement with Qwen:', error.message);
      
      // Try to detect issues even if main analysis fails
      let detectedIssues: string[] = [];
      try {
        detectedIssues = await this.detectIssues(modified || original);
      } catch (e) {
        console.error('Failed to detect issues in fallback:', e);
      }

      // Provide fallback
      const rawAnalysis = `Qwen model evaluation failed. Scores unavailable.

The requirement should be manually reviewed for:
- Ambiguity: Are there multiple interpretations?
- Vagueness: Are terms clearly defined?
- Verifiability: Can this requirement be tested?
- Completeness: Are all necessary details specified?

Error: ${error.message}

Note: preservation_correctness and change_correctness scores must come from Qwen model evaluation and are unavailable when the model fails.`;

      return {
        rawAnalysis,
        parsedAnalysis: {
          preservation_correctness: 0, // 0 indicates Qwen model failure
          change_correctness: 0, // 0 indicates Qwen model failure
          detected_issues: detectedIssues,
          comments: ['Warning: Qwen model evaluation failed. Scores are unavailable.']
        },
      };
    }
  }
}